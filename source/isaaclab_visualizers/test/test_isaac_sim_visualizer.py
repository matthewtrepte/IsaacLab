# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Isaac Sim 7 viewer visualizer: registration, its scene stage and the host's writes."""

from __future__ import annotations

import numpy as np
import pytest
from isaaclab_visualizers.isaac_sim import IsaacSimBackendCfg
from isaaclab_visualizers.isaac_sim.isaac_sim_host_stage import HostStage, SceneWriter

from pxr import Sdf

from isaaclab.scene_data.scene_data_provider import REQUIRES_STAGE_AND_MODEL
from isaaclab.visualizers.visualizer_cfg import VISUALIZER_TYPES

_SCENE = """#usda 1.0
def "World"
{
    def "envs"
    {
        def Xform "env_0"
        {
            def Cube "box"
            {
            }
        }
    }
    over "ground"
    {
    }
}
"""


def test_isaac_sim_is_registered():
    assert "isaac_sim" in VISUALIZER_TYPES
    assert REQUIRES_STAGE_AND_MODEL["isaac_sim"] == (True, False)
    cfg = IsaacSimBackendCfg()
    assert cfg.visualizer_type == "isaac_sim"
    # The viewer's cameras render on a scene stage the host creates, so its panes read what Isaac Lab renders.
    assert cfg.renderer_cfg.stage_provider == "isaaclab_visualizers.isaac_sim.isaac_sim_host_stage:host_stage"


def test_population_types_typeless_containers():
    layer = Sdf.Layer.CreateAnonymous(".usda")
    layer.ImportFromString(HostStage.prepare_population(_SCENE))
    assert layer.GetPrimAtPath("/World").typeName == "Xform"
    assert layer.GetPrimAtPath("/World/envs").typeName == "Xform"
    assert layer.GetPrimAtPath("/World/envs/env_0/box").typeName == "Cube"
    # An over authors no prim of its own, so it keeps no type.
    assert not layer.GetPrimAtPath("/World/ground").typeName


def test_population_of_typed_scene_is_unchanged():
    scene = '#usda 1.0\ndef Xform "World"\n{\n}\n'
    assert HostStage.prepare_population(scene) == scene


def test_viewer_camera_moves_go_to_the_sensor():
    moved = []
    writer = SceneWriter(scene=None, cameras={"/World/viewer_camera_0_0": moved.append})
    pose = np.eye(4)
    pose[3, :3] = (1.0, 2.0, 3.0)
    writer.write(["/World/viewer_camera_0_0"], "omni:xform", pose.reshape(1, 16))
    assert len(moved) == 1
    np.testing.assert_allclose(moved[0], pose)


def test_host_stage_holds_isaac_lab_population():
    pytest.importorskip("isaacsim.context.usd")
    from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

    from isaacsim.common.interfaces import Query

    provider = "isaaclab_visualizers.isaac_sim.isaac_sim_host_stage:host_stage"
    backend = OvstageBackend(OvstageBackendCfg(stage_provider=provider))
    try:
        assert isinstance(backend.stage_owner, HostStage)
        backend.populate(_SCENE)
        context = backend.stage_owner.context
        # Population follows the host context's own first ordinal, and the host reads Isaac Lab's prims.
        assert backend.ordinal > 2
        assert "/World" in context.read(Query("/"), "usd-children")[0]
        assert context.read(Query("/World/envs"), "usd-prim-type")[0] == "Xform"
        assert context.read(Query("/World/envs"), "usd-children")[0] == ["/World/envs/env_0"]
    finally:
        backend.close()


class _FakeAsset:
    """Root poses per environment and the writes the viewer's gizmo makes, in Isaac Lab's (x, y, z, w) order."""

    def __init__(self, prim_path: str, poses):
        import torch

        self.cfg = type("Cfg", (), {"prim_path": prim_path})()
        self._poses = torch.tensor(poses, dtype=torch.float32)
        self.data = type("Data", (), {})()
        self.data.root_link_pose_w = type("Pose", (), {"torch": self._poses})()
        self.data.root_link_vel_w = type("Velocity", (), {"torch": torch.zeros((len(poses), 6))})()
        self.velocities = []

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids):
        self._poses[env_ids.long()] = root_pose

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.velocities.append((env_ids.tolist(), root_velocity.tolist()))


def _backend():
    from types import SimpleNamespace

    from isaaclab_visualizers.isaac_sim.isaac_sim_manipulation import AssetOwners, AssetPoseBackend

    robot = _FakeAsset("/World/envs/env_.*/Robot", [[0, 0, 0, 0, 0, 0, 1], [5, 0, 0, 0, 0, 0, 1]])
    owners = AssetOwners(SimpleNamespace(articulations={"robot": robot}, rigid_objects={}))
    forwards = []
    bodies = ["/World/envs/env_0/Robot/pelvis", "/World/envs/env_1/Robot/pelvis"]
    roots = ["/World/envs/env_0/Robot", "/World/envs/env_1/Robot"]
    return robot, owners, AssetPoseBackend(owners, bodies, roots, lambda: forwards.append(1)), forwards


def test_any_prim_of_an_asset_resolves_to_its_instance():
    _, owners, backend, _ = _backend()
    owner = owners("/World/envs/env_1/Robot/torso_link/visuals/mesh_0")
    assert owner.env_id == 1 and owner.root == "/World/envs/env_1/Robot"
    assert owners("/World/ground") is None
    # The selection manipulation reads body world matrices and moves instance roots.
    assert {"/World/envs/env_1/Robot/pelvis", "/World/envs/env_1/Robot"} <= backend.paths


def test_pose_backend_places_the_instance_at_rest_and_restores_it():
    robot, _, backend, forwards = _backend()
    root = "/World/envs/env_1/Robot"
    captured = backend.capture(root)
    np.testing.assert_allclose(captured[0], [[5, 0, 0]])
    turn = np.array([[0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]])
    backend.hold(root, np.array([[5.0, 0.0, 1.0]]), turn, view=None)
    backend.flush(view=None)
    np.testing.assert_allclose(robot._poses[1].numpy(), [5, 0, 1, *turn[0]], atol=1e-6)
    np.testing.assert_allclose(robot._poses[0].numpy(), [0, 0, 0, 0, 0, 0, 1])
    assert robot.velocities[-1] == ([1], [[0.0] * 6]) and forwards == [1]
    backend.restore(root, captured, view=None)
    np.testing.assert_allclose(robot._poses[1].numpy(), [5, 0, 0, 0, 0, 0, 1])


def test_only_instance_roots_are_moved():
    _, _, backend, _ = _backend()
    with pytest.raises(ValueError, match="asset instances"):
        backend.hold("/World/envs/env_1/Robot/pelvis", np.zeros((1, 3)), np.array([[0, 0, 0, 1.0]]), view=None)
    with pytest.raises(ValueError, match="asset instances"):
        backend.capture("/World/ground")
