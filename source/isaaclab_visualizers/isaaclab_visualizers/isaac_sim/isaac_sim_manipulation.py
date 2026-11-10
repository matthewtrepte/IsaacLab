# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Move and rotate Isaac Lab assets with the Isaac Sim 7 viewer's transform gizmo.

The gestures run through Isaac Sim's own live manipulation: its selection manipulation moves a group rigidly around
the selection's pivot, and its native controller leases each moved body, orders the gesture's commands and restores
the starting pose on cancel. Isaac Sim moves free rigid bodies through Isaac Sim physics; here Isaac Lab owns the
physics, so :class:`AssetPoseBackend` supplies the poses through Isaac Lab's assets instead: selecting any prim of an
articulation or a rigid object moves that asset instance, by writing its root link pose at rest and updating the
kinematics so the change shows while the simulation is paused. Prims that no Isaac Lab asset owns (terrain, lights,
sensors) are not movable, and scaling is not supported, as in Isaac Sim's live manipulation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
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
                if match:
                    env = re.search(r"/env_(\d+)(/|$)", path)
                    owner = Owner(asset, int(env.group(1)) if env else 0, match.group(1))
                    break
            self._cache[path] = owner
        return self._cache[path]


class AssetPoseBackend:
    """Isaac Sim's live-manipulation pose backend, applied to Isaac Lab's asset instances.

    Implements ``isaacsim.viewports.authoring.PoseBackend`` (``capture``, ``hold``, ``restore``, ``flush``) and the
    ``paths`` and ``world`` the host's selection manipulation reads. Isaac Lab drains the gestures on its own thread, so
    these are ordinary Isaac Lab state writes between simulation steps.

    Args:
        owners: Resolves prims to Isaac Lab asset instances.
        bodies: Prim paths whose live world matrices the stage holds (Isaac Lab's rigid bodies); with the asset roots
            they form :attr:`paths`, which the selection manipulation uses to read world matrices.
        roots: Every asset instance root, the paths a gesture moves.
        forward: Updates kinematics after pose writes without stepping physics.
    """

    def __init__(self, owners: AssetOwners, bodies: Iterable[str], roots: Iterable[str], forward: Callable[[], None]):
        self._owners = owners
        self._forward = forward
        self.paths = frozenset((*bodies, *roots))
        self.world: Any = None
        """The host world, whose ``simulation.fail`` the selection manipulation reports unrecoverable errors to."""
        self._dirty = False

    def capture(self, path: str) -> tuple[np.ndarray, ...]:
        """Return the instance's root link position ``[1, 3]``, xyzw orientation ``[1, 4]`` and velocity ``[1, 6]``."""
        instance = self._instance(path)
        data = instance.asset.data
        pose = data.root_link_pose_w.torch[instance.env_id].detach().cpu().numpy().astype(np.float64)
        velocity = data.root_link_vel_w.torch[instance.env_id].detach().cpu().numpy().astype(np.float64)
        return pose[None, :3].copy(), pose[None, 3:7].copy(), velocity[None, :].copy()

    def hold(self, path: str, position: np.ndarray, orientation: np.ndarray, view: Any) -> None:
        """Place the instance's root link at rest."""
        self._write(self._instance(path), position, orientation, np.zeros((1, 6)))

    def restore(self, path: str, captured: Any, view: Any) -> None:
        """Restore the captured pose and velocity after a cancelled gesture."""
        position, orientation, velocity = captured
        self._write(self._instance(path), position, orientation, velocity)

    def flush(self, view: Any) -> None:
        """Update kinematics once after this activation's pose writes, so they show while paused."""
        if self._dirty:
            self._dirty = False
            self._forward()

    def _instance(self, path: str) -> Owner:
        instance = self._owners(path)
        if instance is None or instance.root != path:
            raise ValueError(f"Only Isaac Lab asset instances (articulations, rigid objects) can be moved; not {path}")
        return instance

    def _write(self, instance: Owner, position: Any, orientation: Any, velocity: Any) -> None:
        asset = instance.asset
        device = asset.data.root_link_pose_w.torch.device
        pose = np.concatenate([np.asarray(position, float).reshape(3), np.asarray(orientation, float).reshape(4)])
        env_ids = torch.tensor([instance.env_id], dtype=torch.int32, device=device)
        asset.write_root_link_pose_to_sim_index(
            root_pose=torch.tensor(pose[None], dtype=torch.float32, device=device), env_ids=env_ids
        )
        asset.write_root_link_velocity_to_sim_index(
            root_velocity=torch.tensor(np.asarray(velocity, float).reshape(1, 6), dtype=torch.float32, device=device),
            env_ids=env_ids,
        )
        self._dirty = True


def _stage_world_matrix(view: Any, path: str, bodies: Any = None) -> np.ndarray:
    """Read a prim's world matrix from the stage through any view, as ``world_matrix`` does through a native one."""
    return np.asarray(view.read([path], "omni:fabric:worldMatrix"), dtype=np.float64).reshape(4, 4)


def asset_manipulation(
    world: Any, authoring: Any, owners: AssetOwners, bodies: Iterable[str], forward: Callable[[], None]
) -> Any:
    """Build Isaac Sim's live manipulation of Isaac Lab's asset instances for a host world.

    Args:
        world: The host world.
        authoring: The world's (clone aware) authoring session, which frames selections of Isaac Lab assets.
        owners: Resolves prims to Isaac Lab asset instances.
        bodies: Isaac Lab's rigid body prim paths.
        forward: Updates kinematics without stepping physics.

    Returns:
        Isaac Sim's ``SelectionManipulation`` over a native ``LiveManipulation`` of :class:`AssetPoseBackend`.
    """
    from isaacsim.application.host.impl import selection_manipulation
    from isaacsim.viewports.authoring import LiveManipulation

    # Isaac Sim's selection manipulation reads world matrices with a helper that needs a native simulation view, and
    # Isaac Lab drains the host's commands with a Python one. The stage holds every prim's world matrix, so read it
    # there. Open point for Isaac Sim: a world-matrix helper that accepts any view would make this unnecessary.
    selection_manipulation.world_matrix = _stage_world_matrix
    bodies = tuple(bodies)
    roots = {instance.root for instance in map(owners, bodies) if instance is not None}
    backend = AssetPoseBackend(owners, bodies, roots, forward)
    backend.world = world
    return selection_manipulation.SelectionManipulation(LiveManipulation(backend), authoring)
