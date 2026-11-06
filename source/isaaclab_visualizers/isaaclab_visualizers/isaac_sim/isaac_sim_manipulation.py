# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move and rotate Isaac Lab assets with the Isaac Sim 7 viewer's transform gizmo.

The host's own live manipulation moves free rigid bodies through Isaac Sim physics. Isaac Lab owns the physics here,
so this controller applies the gizmo's poses to Isaac Lab's assets instead: selecting any prim of an articulation or a
rigid object moves that asset in its environment, by writing its root link pose (and zero velocity) and updating the
kinematics so the change shows while the simulation is paused. Prims that no Isaac Lab asset owns (terrain, lights,
sensors) are not movable, and scaling is not supported, as in the host's live manipulation.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
import torch


class Owner(NamedTuple):
    """The Isaac Lab asset instance a prim belongs to."""

    asset: Any
    """The articulation or rigid object."""

    env_id: int
    """Environment index of the instance."""

    root: str
    """Prim path of the instance's root."""


class AssetOwners:
    """Resolves prims to the Isaac Lab asset instances that own them.

    Args:
        scene: Isaac Lab's interactive scene.
    """

    def __init__(self, scene: Any):
        self._patterns = []
        for asset in (*scene.articulations.values(), *scene.rigid_objects.values()):
            pattern = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_.*")
            self._patterns.append((re.compile(f"({pattern})(/.*)?$"), asset))
        self._cache: dict[str, Owner | None] = {}

    def __call__(self, path: str) -> Owner | None:
        """Return the asset instance owning ``path``, or ``None`` if no Isaac Lab asset does."""
        if path not in self._cache:
            owner = None
            for pattern, asset in self._patterns:
                match = pattern.match(path)
                env = re.search(r"/env_(\d+)(/|$)", path) if match else None
                if match:
                    owner = Owner(asset, int(env.group(1)) if env else 0, match.group(1))
                    break
            self._cache[path] = owner
        return self._cache[path]


class MovablePaths:
    """The host's set of live-movable paths: every prim of an Isaac Lab asset."""

    def __init__(self, owners: AssetOwners):
        self._owners = owners

    def __contains__(self, path: object) -> bool:
        return isinstance(path, str) and self._owners(path) is not None


def _rotation(orientation: Any) -> np.ndarray:
    """Return the row-vector rotation matrix of an xyzw quaternion."""
    from pxr import Gf

    x, y, z, w = map(float, orientation)
    return np.asarray(Gf.Matrix3d(Gf.Quatd(w, Gf.Vec3d(x, y, z))))


def _orientation(rotation: np.ndarray) -> list[float]:
    """Return the xyzw quaternion of a row-vector rotation matrix."""
    from pxr import Gf

    quat = Gf.Matrix3d(*rotation.reshape(-1).tolist()).ExtractRotation().GetQuat()
    return [*quat.GetImaginary(), quat.GetReal()]


class AssetManipulation:
    """The host's live manipulation, applied to Isaac Lab's assets.

    Implements what the host's editing routes use: ``supports_groups``, ``backend.paths``, :meth:`apply` and
    :meth:`tick`. Commands run on Isaac Lab's thread when it drains the host's queues, so the writes are ordinary
    Isaac Lab state writes between simulation steps.

    Args:
        owners: Resolves prims to Isaac Lab asset instances.
        authoring: The host's authoring session (clone aware), for the selection's pivot.
        forward: Updates kinematics after pose writes without stepping physics.
    """

    supports_groups = True

    def __init__(self, owners: AssetOwners, authoring: Any, forward: Callable[[], None]):
        self._owners = owners
        self._authoring = authoring
        self._forward = forward
        self.backend = type("Backend", (), {"paths": MovablePaths(owners)})()
        self._groups: dict[tuple[str, str], dict[str, Any]] = {}

    def tick(self, view: Any) -> dict:
        """Report the paths held by an unfinished gesture."""
        return {"held": sorted(root for group in self._groups.values() for root in group["roots"])}

    def apply(self, owner: str, body: dict, view: Any) -> dict:
        """Apply one ordered gesture command (begin, update, heartbeat, end or cancel) to the selection.

        Args:
            owner: Client identity.
            body: Gesture with ``paths``, ``action``, ``sequence`` and, for updates, the pivot's new world pose.
            view: The view the command runs against, for the selection's live world matrices.

        Returns:
            The selection pivot's world position and xyzw orientation.

        Raises:
            ValueError: The selection holds a prim no Isaac Lab asset owns, the gesture expired or scales.
        """
        if "scale" in body:
            raise ValueError("Scaling Isaac Lab assets is unsupported")
        paths = body.get("paths") or [body["path"]]
        key = (owner, body["gesture"])
        action, sequence = body["action"], body["sequence"]
        if action == "begin":
            return self._begin(key, paths, sequence, view)
        group = self._groups.get(key)
        if group is None or group["paths"] != paths:
            raise ValueError("The gesture expired or belongs to another client")
        if sequence <= group["sequence"]:
            raise ValueError("Manipulation sequence must increase")
        group["sequence"] = sequence
        if action == "update":
            self._move(group, np.asarray(body["position"], dtype=float), body["orientation"])
        elif action == "cancel":
            for instance, (position, orientation) in zip(group["owners"], group["initial"]):
                self._write(instance, position, orientation)
            self._forward()
        if action in ("cancel", "end"):
            del self._groups[key]
        return {"position": group["position"], "orientation": group["orientation"]}

    def _begin(self, key: tuple[str, str], paths: list[str], sequence: int, view: Any) -> dict:
        if key in self._groups:
            raise ValueError("This gesture already began; use a new gesture identifier")
        matrices = {}
        for path in paths:
            with_matrix = view.read([path], "omni:fabric:worldMatrix")
            matrices[path] = np.asarray(with_matrix, dtype=float).reshape(4, 4)
        frame = self._authoring.selection_frame(paths, matrices)
        instances: list[Owner] = []
        for root in frame["roots"]:
            instance = self._owners(root)
            if instance is None:
                raise ValueError(f"Only Isaac Lab assets (articulations, rigid objects) can be moved; not {root}")
            if all(instance.asset is not other.asset or instance.env_id != other.env_id for other in instances):
                instances.append(instance)
        held = {root for group in self._groups.values() for root in group["roots"]}
        if held.intersection(instance.root for instance in instances):
            raise ValueError("A selected asset is already being moved")
        pivot = np.asarray(frame["matrix"], dtype=float).reshape(4, 4)
        initial = []
        for instance in instances:
            pose = instance.asset.data.root_link_pose_w.torch[instance.env_id].detach().cpu().numpy().astype(float)
            initial.append((pose[:3], pose[3:7]))
        self._groups[key] = {
            "paths": paths,
            "roots": [instance.root for instance in instances],
            "owners": instances,
            "initial": initial,
            "pivot": pivot,
            "sequence": sequence,
            "position": pivot[3, :3].tolist(),
            "orientation": _orientation(pivot[:3, :3]),
        }
        return {"position": self._groups[key]["position"], "orientation": self._groups[key]["orientation"]}

    def _move(self, group: dict[str, Any], position: np.ndarray, orientation: Any) -> None:
        """Move every asset in the group rigidly with the pivot's new pose."""
        pivot = group["pivot"]
        # Row-vector rotations: the change from the pivot's start to its new orientation.
        delta = pivot[:3, :3].T @ _rotation(orientation)
        for instance, (start, start_orientation) in zip(group["owners"], group["initial"]):
            moved = (start - pivot[3, :3]) @ delta + position
            self._write(instance, moved, _orientation(_rotation(start_orientation) @ delta))
        self._forward()
        group["position"], group["orientation"] = position.tolist(), list(map(float, orientation))

    @staticmethod
    def _write(instance: Owner, position: Any, orientation: Any) -> None:
        """Place an asset instance's root link, at rest."""
        asset = instance.asset
        device = asset.data.root_link_pose_w.torch.device
        pose = torch.tensor([[*map(float, position), *map(float, orientation)]], dtype=torch.float32, device=device)
        env_ids = torch.tensor([instance.env_id], dtype=torch.int32, device=device)
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), dtype=torch.float32, device=device), env_ids=env_ids
        )
