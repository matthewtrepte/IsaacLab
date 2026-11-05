# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 host worlds on Isaac Lab's scene stage.

:class:`HostWorldOpener` is the ``open_world`` callable of the host. Every world it opens, including after the user
stops the scene, reads Isaac Lab's scene stage through the host's stage context and runs no scheduler. Isaac Lab drains
the host's queued commands (camera moves, edits, property inspection) into the stage at its open ordinal, and its
renderer seals them with its next render. The host's transport controls are recorded for Isaac Lab: Pause and Step
gate its loop, and Stop requests an environment reset.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .isaac_sim_host_adapter import (
    BorrowedContext,
    CloneAuthoring,
    HostClock,
    HostData,
    StageView,
    live_inspection_type,
)
from .isaac_sim_host_stage import FrameStore, SceneWriter

if TYPE_CHECKING:
    from isaacsim.application.host import ViewportSpec, World
    from isaacsim.foundation.orchestration import RationalRate

logger = logging.getLogger(__name__)


def _host_world_type() -> type:
    from isaacsim.application.host import World

    class HostWorld(World):
        """Host world whose queued work Isaac Lab drains on its own steps.

        The host loop steps a stopped world while it has pending work. Here only Isaac Lab drains it, so the loop must
        not spin on it; it steps when Isaac Lab requests a redraw after rendering. A simulation step while paused (Step)
        is recorded as a grant for Isaac Lab's loop instead of advancing a clock.
        """

        on_simulation_step: Callable[[], None] | None = None

        def has_pending(self) -> bool:
            return False

        def set_running(self, running: bool) -> None:
            self.running = running

        def step_simulation_frame(self) -> int:
            # While playing, the host steps at its own rate; Isaac Lab runs freely then, so only a Step while paused
            # is a grant.
            if not self.running and self.on_simulation_step is not None:
                self.on_simulation_step()
            self.frames_stepped += 1
            return self.sim_time_ns

    return HostWorld


class HostWorldOpener:
    """Open host worlds on Isaac Lab's scene stage.

    Args:
        scene_file: Isaac Lab's authored scene, exported for the host's authoring session (attribute types and edits).
        context: The host's stage context over Isaac Lab's scene stage (an ``isaacsim.context.usd.UsdContext``).
        writer: Writes the host's edits at Isaac Lab's open ordinal.
        specs: The viewer's viewports, one per Isaac Lab viewer camera.
        frames: Latest frame of each viewport.
        rate: Host frame rate.
        up: Stage up axis as a vector.
        meters_per_unit: Stage units [m].
        copies: Isaac Lab's environment clones as ``(source, targets)``, so the property pane describes clones.
    """

    def __init__(
        self,
        scene_file: Path,
        context: Any,
        writer: SceneWriter,
        specs: Sequence[ViewportSpec],
        frames: FrameStore,
        rate: RationalRate,
        up: tuple[float, float, float],
        meters_per_unit: float,
        copies: Sequence[tuple[str, Sequence[str]]],
    ):
        from isaacsim.viewports.authoring import AuthoringSession

        self.authoring = AuthoringSession(str(scene_file.resolve()))
        self._context = context
        self._writer = writer
        self._specs = tuple(specs)
        self.frames = frames
        self._rate = rate
        self._up = up
        self._meters_per_unit = meters_per_unit
        self._copies = copies
        self._lock = threading.Lock()
        self._world: World | None = None
        self._data: HostData | None = None
        self._activation = 0
        self._opened = 0
        self._step_grants = 0
        self._reset_requested = False
        self._revision: int | None = None
        self.enabled: set[str] = set()
        """Products of the viewports a client currently shows."""

    def drain(self, time_ns: int) -> bool:
        """Apply the host's queued commands at Isaac Lab's open ordinal.

        Args:
            time_ns: Isaac Lab's simulation time [ns].

        Returns:
            Whether a command changed the stage, so the viewports need a new frame.
        """
        with self._lock:
            world, data = self._world, self._data
        if world is None or data is None:
            return False
        data.time_ns = time_ns
        self._activation += 1
        view = StageView(data)
        changed = False
        for source in (world.commands, *world.command_sources):
            try:
                source._capture_activation(view, self._activation)
                changed |= source.snapshot.applied > 0
            except Exception:  # noqa: BLE001 - a viewer command must not stop Isaac Lab's simulation
                logger.exception("An Isaac Sim viewer command failed; continuing.")
        return changed

    def published(self) -> None:
        """Resolve the drained commands' futures once Isaac Lab's renderer has sealed their writes."""
        with self._lock:
            world = self._world
        if world is not None:
            for source in (world.commands, *world.command_sources):
                source._published()

    def __call__(self, scene: str, wake: Callable[[], None]) -> World:
        """Open a host world on Isaac Lab's scene stage."""
        from isaacsim.application.host import AuthoringSync, SceneCommands
        from isaacsim.foundation.orchestration import RationalRate
        from isaacsim.viewports.adapters import RendererOutput

        context = BorrowedContext(self._context)
        data = HostData(context, self._writer)
        commands = SceneCommands(rate=self._rate, on_submit=wake)
        world = _host_world_type()(
            context,
            data,
            commands,
            self._rate,
            scene,
            clock=HostClock(data),
            simulation_participants=[],
            up=self._up,
            meters_per_unit=self._meters_per_unit,
            on_external_submit=wake,
        )
        world.on_simulation_step = self._grant_step
        world.renderer_name = "ovrtx"
        world.authoring = CloneAuthoring(self.authoring, self._copies)
        world.authoring_revision = self.authoring.revision
        world.authoring_sync = AuthoringSync(self.authoring.stage)
        # Property inspection supplies the selected prims' live world poses; Isaac Lab drains it with the commands.
        inspection = SceneCommands(rate=RationalRate(10), on_submit=wake)
        world.inspection = live_inspection_type()(inspection, set(), context)
        world.command_sources.append(inspection)
        frames = self.frames
        world.renderer_output_factory = lambda path, product: RendererOutput(read=lambda: frames.latest(product))
        self._attach_viewports(world)
        with self._lock:
            self._world, self._data = world, data
            self._opened += 1
            # The host reopens its world when the user stops the scene, and also to apply authored edits while playing.
            # Only Stop resets Isaac Lab's environments; an edit changes the authoring revision.
            revision = self.authoring.revision
            self._reset_requested = self._opened > 1 and revision == self._revision
            self._revision = revision
        return world

    def _attach_viewports(self, world: World) -> None:
        """Offer Isaac Lab's viewer cameras as the host's viewports; they are declared at launch, not created here."""
        products = {spec.product for spec in self._specs}

        def create_viewport(spec: ViewportSpec) -> Future:
            done: Future = Future()
            if spec.product in products:
                done.set_result(None)
            else:
                done.set_exception(ValueError(f"Isaac Lab declares {len(products)} viewer camera(s) at launch"))
            return done

        def set_enabled(product: str, enabled: bool) -> None:
            (self.enabled.add if enabled else self.enabled.discard)(product)

        world.viewport_specs = self._specs
        world.create_viewport = create_viewport
        world.remove_viewport = lambda spec: set_enabled(spec.product, False)
        world.set_render_product_enabled = set_enabled

    def _grant_step(self) -> None:
        with self._lock:
            self._step_grants += 1

    def take_step(self) -> bool:
        """Consume one step the user requested while paused; return whether one was pending."""
        with self._lock:
            if self._step_grants:
                self._step_grants -= 1
                return True
            return False

    def clear_steps(self) -> None:
        """Drop steps granted while the host was playing; they only matter while it is paused."""
        with self._lock:
            self._step_grants = 0

    def take_reset(self) -> bool:
        """Consume a reset that the host's Stop requested; return whether one was pending."""
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
            return requested
