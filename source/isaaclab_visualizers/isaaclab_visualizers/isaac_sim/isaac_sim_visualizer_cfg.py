# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Isaac Sim 7 viewer visualizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from isaaclab_ov.renderers import OVRTXRendererCfg

from isaaclab.utils import configclass
from isaaclab.visualizers.visualizer_cfg import VisualizerBackendCfg

if TYPE_CHECKING:
    from .isaac_sim_visualizer import IsaacSimVisualizer


@configclass
class IsaacSimBackendCfg(VisualizerBackendCfg):
    """Configuration for the Isaac Sim 7 viewer attached to an Isaac Lab simulation.

    Isaac Lab keeps its own physics and loop. The viewer shows Isaac Lab's own OVRTX scene: launch preparation declares
    the viewer's perspective cameras as ordinary scene cameras on :attr:`renderer_cfg`, whose scene stage the Isaac Sim
    7 host creates (:func:`~.isaac_sim_host_stage.host_stage`) so its stage tree and property panes read it. Isaac Lab
    populates that stage and owns its ordinals; the host runs no scheduler. The shared
    :class:`~isaaclab.visualizers.visualizer_cfg.VisualizerCfg` supplies the viewport size (``window.size``), the frame
    rate (``window.fps``) and ``headless``, which serves the viewer without a client.
    """

    class_type: type[IsaacSimVisualizer] | str = "{DIR}.isaac_sim_visualizer:IsaacSimVisualizer"
    """Visualizer implementation class."""

    visualizer_type: str = "isaac_sim"
    """Type identifier for the Isaac Sim 7 viewer."""

    launcher_type: ClassVar[str] = "isaaclab_ov.app:OvrtxLauncher"
    """Registers the OVRTX runtime before the stage is created."""

    renderer_cfg: OVRTXRendererCfg = OVRTXRendererCfg(
        stage_provider="isaaclab_visualizers.isaac_sim.isaac_sim_host_stage:host_stage"
    )
    """Renderer of the viewer's cameras. Its stage provider lets the host's panes read Isaac Lab's scene stage.

    Scene cameras with the same configuration share the instance and its stage.
    """

    render_settings: dict[str, Any] = dict()
    """RTX attributes to author on the viewer cameras' render products, as ``{name: (usd_type_name, value)}``."""

    camera_controls: Literal["newton", "isaac_sim"] = "newton"
    """Viewport camera bindings.

    ``"newton"`` matches Isaac Lab's Newton viewer: left-drag looks, middle-drag orbits, Shift+middle-drag pans and
    Ctrl+middle-drag dollies; a left click still selects. ``"isaac_sim"`` keeps the client's Kit-style bindings:
    Alt+left-drag orbits, middle-drag pans and right-drag looks. See :mod:`.isaac_sim_camera_controls`.
    """

    ui: Literal["desktop", "browser"] = "desktop"
    """Where the viewer is presented: the Electron window attached to the host, or a logged browser URL."""

    presentation: Literal["auto", "gpu", "http"] = "http"
    """How the desktop client receives frames. ``"http"`` sends encoded CPU images, which Isaac Lab's frames are."""

    port: int = 0
    """Port of the host's HTTP service; zero picks a free one."""

    app_client_root: str | None = None
    """Directory with the built ``sim_viewer`` application client: ``renderer/`` (``index.js``,
    ``manifest.js``) and, for ``ui="desktop"``, ``main.cjs``.

    The renderer supplies the stage, property and content panes. ``None`` uses the
    ``ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT`` environment variable, or serves only the shared shell when it is unset.
    """
