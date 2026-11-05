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
