# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualizer that displays an Isaac Lab simulation in the Isaac Sim 7 viewer application."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from pxr import UsdGeom

from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_matrix
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, VisualizerCfg

from .isaac_sim_visualizer_cfg import IsaacSimBackendCfg

if TYPE_CHECKING:
    from isaaclab.sensors.camera import Camera
    from isaaclab.visualizers.image_view import ImageView

logger = logging.getLogger(__name__)

_VIEWER_CAMERA_PREFIX = "/World/viewer_camera_"
"""Prim path prefix of the perspective cameras launch preparation declares for viewers."""


class IsaacSimVisualizer(BaseVisualizer):
    """Isaac Sim 7 viewer application on Isaac Lab's own OVRTX scene.

    Launch preparation declares the viewer's perspective cameras as ordinary scene cameras on
    :attr:`IsaacSimBackendCfg.renderer_cfg`. That renderer's scene stage is created by the host's stage context, so the
    host's stage tree and property panes read the stage Isaac Lab renders. Each frame, Isaac Lab applies the host's
    queued commands at its open ordinal, renders the viewer cameras, and hands their images to the host, which streams
    them to its client. Isaac Lab keeps its own loop and physics; the host runs no scheduler.
    """

    def __init__(self, cfg: VisualizerCfg):
        """Initialize visualizer state.

        Args:
            cfg: Shared visualizer settings whose ``backend`` is an :class:`IsaacSimBackendCfg`.
        """
        super().__init__(cfg)
        self.backend_cfg: IsaacSimBackendCfg = cfg.backend
        if self.backend_cfg.app_client_root is None:
            self.backend_cfg.app_client_root = os.environ.get("ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT")
        self._resources = contextlib.ExitStack()
        self._host = None
        self._desktop = None
        self._opener = None
        self._viewports: dict[str, ImageView] = {}
        """Viewer camera image view per host render product."""
        self._shown: set[str] = set()
        """Products whose current frame the host has; a viewport that becomes visible gets one first."""
        self._marker_revision = -1
        self._next_render = 0.0
        self._running = False
        self.renders = 0
        self.timings = {"publish": 0.0, "steps": 0}
        """Accumulated wall time [s] of the viewer's work in :meth:`step`, for benchmarking."""

    @property
    def client_url(self) -> str:
        """Address that opens the viewer in a browser, including the launch token."""
        return f"{self._host.url}/index.html?entry=index.js&token={self._host.token}"

    def initialize(self, sim: SimulationContext, *, cameras: list[PerspectiveCameraCfg | Camera]) -> None:
        """Start the Isaac Sim 7 host on Isaac Lab's scene stage and open its client.

        Args:
            sim: Simulation owner that provides the authored stage and scene data.
            cameras: Resolved perspective settings and borrowed scene sensors, in display order.
        """
        if self._is_initialized:
            logger.debug("[IsaacSimVisualizer] initialize() called while already initialized.")
            return
        if not self.cfg.headless and self.backend_cfg.ui == "desktop" and self.backend_cfg.app_client_root is None:
            raise ValueError(
                "IsaacSimBackendCfg.app_client_root (or ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT) is required for"
                " ui='desktop'."
            )
        super().initialize(sim, cameras=cameras)
        views = [
            view
            for view in self._image_views
            if view.camera is not None and view.camera.cfg.prim_path.startswith(_VIEWER_CAMERA_PREFIX)
        ]
        if not views:
            raise ValueError(
                "The Isaac Sim viewer shows the perspective cameras that launch preparation declares; pass the"
                " environment config to launch_simulation so they are declared before scene construction."
            )
        self._viewports = {f"/IsaacLabViewer/Viewport_{index}": view for index, view in enumerate(views)}
        renderer = sim.get_or_create_backend(self.backend_cfg.renderer_cfg)
        if self.cfg.enable_markers:
            renderer.marker_groups = sim.vis_marker_registry.get_groups().values()
        self._start_host(sim, renderer)
        self._log_initialization_table(
            logger=logger,
            title="IsaacSimVisualizer Configuration",
            rows=[
                ("ui", "headless" if self.cfg.headless else self.backend_cfg.ui),
                ("viewports", len(self._viewports)),
                ("fps", self.cfg.window.fps),
            ],
        )
        self._running = True
        self._is_initialized = True

    def _start_host(self, sim: SimulationContext, renderer) -> None:
        """Run the Isaac Sim 7 application host, with its panes and transport controls, on Isaac Lab's scene stage."""
        from sim_viewer.cli import client_roots_for_backend

        from isaacsim.application.host import AppHost, ViewportSpec
        from isaacsim.foundation.orchestration import RationalRate
        from isaacsim.viewports.presentation import FrameEncoder

        from .isaac_sim_host_stage import FrameStore, HostStage, SceneWriter
        from .isaac_sim_host_world import HostWorldOpener

        scene = renderer.scene
        if not isinstance(getattr(scene, "stage_owner", None), HostStage):
            raise RuntimeError("The viewer's renderer has no scene stage from the host; check its stage_provider.")
        directory = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_isaac_sim_")))
        scene_file = directory / "scene.usda"
        # The host's authoring session reads attribute types and authors edits from Isaac Lab's authored layer.
        sim.stage.GetRootLayer().Export(str(scene_file))
        up = (0.0, 1.0, 0.0) if UsdGeom.GetStageUpAxis(sim.stage) == UsdGeom.Tokens.y else (0.0, 0.0, 1.0)
        width, height = self.cfg.window.size
        specs = [
            ViewportSpec(index=index, product=product, camera=view.camera.cfg.prim_path, resolution=(width, height))
            for index, (product, view) in enumerate(self._viewports.items())
        ]
        cameras = {
            view.camera.cfg.prim_path: (lambda matrix, view=view: self._move_camera(view, matrix))
            for view in self._viewports.values()
        }
        self._opener = HostWorldOpener(
            scene_file,
            scene.stage_owner.context,
            SceneWriter(scene, cameras),
            specs,
            FrameStore(),
            RationalRate(int(self.cfg.window.fps)),
            up,
            UsdGeom.GetStageMetersPerUnit(sim.stage),
            scene.clone_copies,
            renderer,
        )
        client_roots = client_roots_for_backend("web")
        if self.backend_cfg.app_client_root is not None:
            client_roots = [("/app/", Path(self.backend_cfg.app_client_root) / "renderer"), *client_roots[1:]]
        if self.backend_cfg.camera_controls == "newton":
            from .isaac_sim_camera_controls import newton_controls_client

            client = Path(self._resources.enter_context(tempfile.TemporaryDirectory(prefix="isaaclab_client_")))
            client_roots = [
                (prefix, newton_controls_client(root, client) or root) if prefix == "/" else (prefix, root)
                for prefix, root in client_roots
            ]

        def viewport(index: int) -> ViewportSpec:
            if index >= len(specs):
                raise ValueError(f"Isaac Lab declares {len(specs)} viewer camera(s) at launch")
            return specs[index]

        from isaacsim.common.interfaces import Query

        # Hide the stage's bookkeeping and the renderer's products from the stage tree; the scene is under /World.
        roots = scene.stage_owner.context.read(Query("/"), "usd-children")[0]
        hidden = tuple(root for root in roots if root not in ("/World", "/physicsScene"))
        self._host = AppHost(
            open_world=self._opener,
            hidden_roots=hidden,
            scene=str(scene_file),
            viewports=[],
            viewport_factory=viewport,
            encoder=FrameEncoder.create("auto"),
            presentation=self.backend_cfg.presentation,
            renderer_name="ovrtx",
            world_up=up,
            host="127.0.0.1",
            port=self.backend_cfg.port,
            client_roots=client_roots,
            content_writes=True,
            stage_writes=True,
        )
        self._resources.callback(self._host.stop)
        self._host.start()
        # Isaac Lab is already running, so the viewer's transport controls start in Playing.
        self._play()
        if not self.cfg.headless and self.backend_cfg.ui == "browser":
            self._log_viewer_url("IsaacSimVisualizer", self.client_url)
        elif not self.cfg.headless:
            self._open_desktop()

    def _open_desktop(self) -> None:
        """Open the Electron window attached to the host that this process runs."""
        from isaacsim.application.client import start_desktop

        self._desktop = start_desktop(
            Path(self.backend_cfg.app_client_root) / "main.cjs",
            environment={"SIM_VIEWER_ATTACH_URL": self._host.url, "SIM_VIEWER_ATTACH_TOKEN": self._host.token},
        )
        self._resources.callback(self._desktop.close)

    def _move_camera(self, view: ImageView, local: np.ndarray) -> None:
        """Move a viewer camera sensor to the local transform the host's navigation wrote.

        Args:
            view: The camera's image view.
            local: Row-major camera transform relative to the camera's parent, which is the identity-posed ``/World``.
        """
        camera = view.camera
        position = torch.as_tensor(local[3, :3], dtype=torch.float32, device=camera.device).unsqueeze(0)
        # USD matrices multiply row vectors, so the rotation's columns are the rows of the upper 3x3 block.
        rotation = torch.as_tensor(local[:3, :3].T, dtype=torch.float32, device=camera.device)
        camera.set_world_poses(position, quat_from_matrix(rotation).unsqueeze(0), convention="opengl")
        view.invalidate()

    def _render_frame(self) -> None:
        """Apply the host's commands and render the shown viewports, at most ``window.fps`` times a second."""
        now = time.monotonic()
        if now < self._next_render:
            return
        self._next_render = now + 1.0 / self.cfg.window.fps
        time_ns = int(self._sim_time * 1e9)
        changed = self._opener.drain(time_ns)
        if self._opener.redraw:
            # A pick or a new selection outline needs a render even when the scene is unchanged.
            self._opener.redraw = False
            changed = True
        if changed:
            for view in self._viewports.values():
                view.invalidate()
        if self.cfg.enable_markers and self._marker_revision != self._sim.vis_marker_registry.revision:
            self._marker_revision = self._sim.vis_marker_registry.revision
            for view in self._viewports.values():
                view.invalidate()
        frame_id = self._sim.get_physics_step_count()
        published = False
        self._shown &= self._opener.enabled
        for product, view in self._viewports.items():
            if product not in self._opener.enabled:
                continue
            fresh = view.frame.timestamp != frame_id or product not in self._shown
            image = view.read(frame_id)
            if image is None or not fresh:
                continue
            self._shown.add(product)
            self._opener.frames.publish(product, np.ascontiguousarray(image.numpy()), time_ns)
            published = True
        # A render seals the commands' writes; without one they stay at the open ordinal until the next.
        if published:
            self.renders += 1
            self._opener.published()
            self._host.simulation.request_redraw()

    def _play(self) -> None:
        """Put the viewer's transport controls in Playing."""
        from isaacsim.application.host import Command, TimeTarget

        self._host.inbox.submit(Command(verb="play", args={}, time_target=TimeTarget.immediate()))

    def is_training_paused(self) -> bool:
        """Return whether the viewer's Pause holds Isaac Lab's loop; each Step lets one simulation step through."""
        if self._host is None or self._opener is None:
            return False
        if self._host.simulation.run_state.value != "paused":
            self._opener.clear_steps()
            return False
        return not self._opener.take_step()

    def supports_markers(self) -> bool:
        """Return whether visualization markers are drawn."""
        return self.cfg.enable_markers

    def step(self, dt: float) -> None:
        """Apply the viewer's commands and stream its viewports.

        Args:
            dt: Simulation time-step in seconds.
        """
        if not self._is_initialized or not self._running:
            return
        self._sim_time += dt
        start = time.perf_counter()
        if self._opener.take_reset():
            # The viewer's Stop resets the environments, which then keep running.
            self._reset_requested = True
            self._play()
        self._render_frame()
        if dt == 0.0:
            # Isaac Lab calls this in a loop while the viewer is paused; keep serving the viewer without spinning.
            time.sleep(0.002)
        self.timings["publish"] += time.perf_counter() - start
        self.timings["steps"] += 1

    def is_running(self) -> bool:
        """Return whether the viewer is still open; closing the desktop window ends it."""
        if self._desktop is not None and self._desktop.returncode is not None:
            self._running = False
        return self._running

    def close(self) -> None:
        """Stop the host and remove the exported scene; the scene cameras keep their renderer."""
        if self._is_closed:
            return
        try:
            self._resources.close()
        finally:
            self._running = False
            self._host = self._desktop = self._opener = None
            self._viewports.clear()
            super().close()
