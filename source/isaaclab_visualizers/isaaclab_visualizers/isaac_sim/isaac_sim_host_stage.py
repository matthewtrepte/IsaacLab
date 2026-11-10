# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Lab's scene stage, created through the Isaac Sim 7 host's stage context.

The host's stage tree and property panes read a stage through an ``isaacsim.context.usd.UsdContext``, which can only
read a stage it created itself. :func:`host_stage` is the :attr:`~isaaclab_ov.renderers.OVRTXRendererCfg.stage_provider`
the viewer selects: the context creates an empty OVStage, and Isaac Lab borrows it, populates it and owns its ordinals
as it does any scene stage.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import Any, NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

_MATRIX_ATTRIBUTES = frozenset({"omni:xform", "omni:fabric:worldMatrix", "omni:fabric:localMatrix", "worldMatrix"})


class HostStage(NamedTuple):
    """A stage the host's context created and Isaac Lab borrows."""

    stage: Any
    """Isaac Lab's borrowed view of the native stage (an ``ovstage.Stage``)."""

    context: Any
    """The host's stage context that created it (an ``isaacsim.context.usd.UsdContext``)."""

    @staticmethod
    def prepare_population(usda: str) -> str:
        """Give typeless prims the ``Xform`` type before Isaac Lab populates the stage.

        Isaac Lab's spawners leave containers such as ``/World`` and ``/World/envs`` typeless, and the host's stage
        context lists only typed prims as children, so its stage tree would not show them.
        """
        from pxr import Sdf

        layer = Sdf.Layer.CreateAnonymous(".usda")
        layer.ImportFromString(usda)
        typeless = []

        def visit(path: Sdf.Path) -> None:
            spec = layer.GetPrimAtPath(path)
            if spec is not None and spec.specifier == Sdf.SpecifierDef and not spec.typeName:
                typeless.append(spec)

        layer.Traverse(Sdf.Path.absoluteRootPath, visit)
        if not typeless:
            return usda
        for spec in typeless:
            spec.typeName = "Xform"
        return layer.ExportToString()


@contextlib.contextmanager
def host_stage(name: str) -> Iterator[HostStage]:
    """Create an empty stage through the host's stage context for Isaac Lab to populate.

    Args:
        name: Isaac Lab's name for the stage, for diagnostics.

    Yields:
        The borrowed stage and its context, valid until the context manager exits.
    """
    from isaaclab_ov.stage import create_ovstage

    from isaacsim.context.usd import Backend, make_usd_context
    from isaacsim.usd.ovstage import borrow_stage

    # ovstage fixes its hierarchy model per process at the first stage; claim Isaac Lab's GPU-incremental model before
    # the host's context creates its stage, so OVRTX reads world transforms on the device.
    with create_ovstage(f"{name}.anchor"):
        context = make_usd_context(Backend.OVSTAGE)
        context.create_from_string('#usda 1.0\n(\n    upAxis = "Z"\n    metersPerUnit = 1\n)\n')
        stage = borrow_stage(context.stage_pointer)
        try:
            yield HostStage(stage, context)
        finally:
            stage.destroy()


class SceneWriter:
    """Writes the host's edits into Isaac Lab's scene stage at the ordinal Isaac Lab has open.

    Isaac Lab's renderer seals that ordinal when it next renders, so the edits become visible with Isaac Lab's own
    writes. Writes of a viewer camera's local transform go to the camera sensor instead: Isaac Lab's renderer writes
    its cameras' poses from their sensors, so the sensor is where the viewer's navigation has to land.

    Args:
        scene: Isaac Lab's scene stage (an ``isaaclab_ov.stage.OvstageBackend``).
        cameras: Viewer camera prim paths, each with a function that moves its sensor to a local transform.
    """

    def __init__(self, scene: Any, cameras: dict[str, Callable[[np.ndarray], None]]):
        self._scene = scene
        self._cameras = cameras

    def write(self, paths: Sequence[str], attribute: str, values: Any) -> None:
        """Write one value per path at the open ordinal."""
        import ovstage

        paths = list(paths)
        array = np.asarray(values)
        if attribute == "omni:xform" and paths and all(path in self._cameras for path in paths):
            for path, matrix in zip(paths, np.asarray(array, dtype=np.float64).reshape(len(paths), 4, 4)):
                self._cameras[path](matrix)
            return
        kwargs = {}
        if array.dtype.kind in "USO":
            # Text values, such as ``visibility``, are stored as interned tokens.
            array = np.asarray([self._scene.paths.intern_token(str(value)) for value in array.reshape(-1)], np.uint64)
            kwargs["semantic"] = ovstage.AttributeSemantic.TOKEN_ID
        elif attribute in _MATRIX_ATTRIBUTES:
            array = np.ascontiguousarray(array, dtype=np.float64).reshape(len(paths), 16)
            kwargs["semantic"] = ovstage.AttributeSemantic.MATRIX
        elif array.ndim:
            array = np.ascontiguousarray(array).reshape(len(paths), *array.shape[1:])
        with self._scene.query(paths) as query:
            self._scene.stage.write_attribute(
                query, attribute, ordinal=self._scene.ordinal, tensors=array, is_array=False, **kwargs
            ).wait()

    def clone(self, source: str, targets: Sequence[str]) -> None:
        """Clone a subtree at the open ordinal."""
        self._scene.stage.clone(source, list(targets), ordinal=self._scene.ordinal)

    def remove(self, paths: Sequence[str]) -> None:
        """Delete prim subtrees at the open ordinal."""
        with self._scene.query(list(paths)) as query:
            self._scene.stage.delete_attributes(query, [], ordinal=self._scene.ordinal).wait()


class GizmoSelections:
    """Each client's selection, which the host reports for its transform gizmo.

    The host's own gizmo capture also computes the overlay calibration; here Isaac Lab attaches it to each frame it
    publishes, so this only records the selections.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._selections: dict[str, tuple[str, ...]] = {}

    def select(self, paths: Sequence[str], *, owner: str = "") -> None:
        """Record a client's selection; an empty or oversized one has no gizmo."""
        with self._lock:
            if 0 < len(paths) <= 128:
                self._selections[owner] = tuple(paths)
            else:
                self._selections.pop(owner, None)

    def selections(self) -> tuple[tuple[str, ...], ...]:
        """Return the distinct selections of all clients."""
        with self._lock:
            return tuple(dict.fromkeys(self._selections.values()))


class FrameStore:
    """Latest viewer frame per render product, in the shape the host's renderer outputs read."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frames: dict[str, Any] = {}
        self._ids = itertools.count(1)

    def publish(
        self,
        product: str,
        pixels: np.ndarray,
        time_ns: int,
        *,
        generation: int = 0,
        camera: Any = None,
        transforms: Sequence[tuple[str, Sequence[float]]] = (),
    ) -> None:
        """Make an RGBA ``[H, W, 4]`` uint8 image the latest frame of ``product``.

        Args:
            product: Render product the image is of.
            pixels: The image.
            time_ns: Simulation time [ns] of the image.
            generation: The host world's generation the image belongs to.
            camera: The camera's calibration for the image (an ``isaacsim.viewports.manager.CameraSnapshot``), which
                the client needs to draw its transform gizmo over the image.
            transforms: World matrices of the selected prims and selection pivots, keyed by path or pivot key.
        """
        frame_id = next(self._ids)
        frame = SimpleNamespace(
            pixels=pixels,
            memory="cpu",
            frame_id=frame_id,
            time_ns=time_ns,
            source="isaaclab-ovrtx",
            generation=generation,
            input_ordinal=0,
            output_ordinal=0,
            capture_id=frame_id,
            timeline="participant",
            transforms=tuple((key, tuple(map(float, matrix))) for key, matrix in transforms),
            camera=camera,
            depth=None,
        )
        with self._lock:
            self._frames[product] = frame

    def latest(self, product: str) -> Any:
        with self._lock:
            return self._frames.get(product)
