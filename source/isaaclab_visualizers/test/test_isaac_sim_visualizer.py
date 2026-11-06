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
        self.velocities = []

    def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids):
        self._poses[env_ids.long()] = root_pose

    def write_root_link_velocity_to_sim_index(self, *, root_velocity, env_ids):
        self.velocities.append((env_ids.tolist(), root_velocity.tolist()))


def _manipulation():
    from types import SimpleNamespace

    from isaaclab_visualizers.isaac_sim.isaac_sim_host_adapter import CloneAuthoring
    from isaaclab_visualizers.isaac_sim.isaac_sim_manipulation import AssetManipulation, AssetOwners

    robot = _FakeAsset("/World/envs/env_.*/Robot", [[0, 0, 0, 0, 0, 0, 1], [5, 0, 0, 0, 0, 0, 1]])
    scene = SimpleNamespace(articulations={"robot": robot}, rigid_objects={})
    owners = AssetOwners(scene)
    forwards = []
    authoring = CloneAuthoring(session=None, copies=(), owners=owners)
    return robot, owners, AssetManipulation(owners, authoring, lambda: forwards.append(1)), forwards


class _PoseView:
    """World matrices of selected prims, as the host's view reads them."""

    def __init__(self, positions: dict[str, tuple[float, float, float]]):
        self._positions = positions

    def read(self, paths, attribute):
        matrix = np.eye(4)
        matrix[3, :3] = self._positions[paths[0]]
        return matrix.reshape(1, 16)


def test_any_prim_of_an_asset_moves_its_instance():
    _, owners, manipulation, _ = _manipulation()
    owner = owners("/World/envs/env_1/Robot/torso_link/visuals/mesh_0")
    assert owner.env_id == 1 and owner.root == "/World/envs/env_1/Robot"
    assert owners("/World/ground") is None
    assert "/World/envs/env_1/Robot/pelvis" in manipulation.backend.paths
    assert "/World/ground" not in manipulation.backend.paths


def test_gizmo_moves_and_turns_the_owning_instance_at_rest():
    robot, _, manipulation, forwards = _manipulation()
    mesh = "/World/envs/env_1/Robot/torso_link/visuals/mesh_0"
    view = _PoseView({mesh: (5.0, 0.0, 1.0)})
    gesture = {"paths": [mesh], "gesture": "g"}
    begun = manipulation.apply("me", {**gesture, "action": "begin", "sequence": 0}, view)
    assert begun["position"] == [5.0, 0.0, 1.0]
    assert manipulation.tick(view)["held"] == ["/World/envs/env_1/Robot"]
    # Raise the pivot 1 m and turn 90 degrees about z: the root, 1 m below the pivot, follows rigidly.
    turn = [0.0, 0.0, float(np.sin(np.pi / 4)), float(np.cos(np.pi / 4))]
    manipulation.apply(
        "me", {**gesture, "action": "update", "sequence": 1, "position": [5.0, 0.0, 2.0], "orientation": turn}, view
    )
    np.testing.assert_allclose(robot._poses[1, :3].numpy(), [5.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(np.abs(robot._poses[1, 3:].numpy()), np.abs(turn), atol=1e-6)
    np.testing.assert_allclose(robot._poses[0].numpy(), [0, 0, 0, 0, 0, 0, 1])
    assert robot.velocities[-1] == ([1], [[0.0] * 6]) and forwards
    manipulation.apply("me", {**gesture, "action": "end", "sequence": 2}, view)
    assert manipulation.tick(view)["held"] == []


def test_cancel_restores_the_starting_pose():
    robot, _, manipulation, _ = _manipulation()
    mesh = "/World/envs/env_1/Robot/pelvis"
    view = _PoseView({mesh: (5.0, 0.0, 0.0)})
    gesture = {"paths": [mesh], "gesture": "g"}
    manipulation.apply("me", {**gesture, "action": "begin", "sequence": 0}, view)
    manipulation.apply(
        "me",
        {**gesture, "action": "update", "sequence": 1, "position": [9.0, 9.0, 9.0], "orientation": [0, 0, 0, 1]},
        view,
    )
    manipulation.apply("me", {**gesture, "action": "cancel", "sequence": 2}, view)
    np.testing.assert_allclose(robot._poses[1].numpy(), [5, 0, 0, 0, 0, 0, 1])


def test_scaling_an_asset_is_rejected():
    _, _, manipulation, _ = _manipulation()
    with pytest.raises(ValueError, match="Scaling"):
        manipulation.apply(
            "me",
            {
                "paths": ["/World/envs/env_0/Robot"],
                "gesture": "g",
                "action": "update",
                "sequence": 1,
                "scale": [2, 2, 2],
            },
            None,
        )
