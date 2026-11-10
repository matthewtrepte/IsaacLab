# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Isaac Sim 7 viewer visualizer backend (prototype), selected with ``--viz isaac-sim``.

The viewer shows Isaac Lab's own OVRTX scene. Its cameras are scene cameras on a shared OVRTX renderer whose scene stage
the Isaac Sim 7 application host creates, so the host's stage tree and property panes read the stage Isaac Lab renders.
Isaac Lab populates that stage, owns its ordinals, and draws body poses and visualization markers through its renderer.
The host supplies its panes, transport controls and frame streaming; it runs no scheduler, and Isaac Lab drives every
update.

Prototype requirements, not yet captured by the package dependencies:

* Isaac Sim 7 ``7.0.0a2.dev55385`` wheels (including ``isaacsim-rendering-ovgl`` and ``isaacsim-rendering-ovrtx``),
  with ``ovstage==0.4.0.1793102128``, ``ovphysx==0.8.0.41080602166`` and ``ovrtx==0.7.0.dev1793102128`` from the
  ``ct-omniverse-pypi-local`` index.
* The ``sim_viewer`` application package from the Isaac Sim source tree on ``PYTHONPATH``, and a built
  ``sim_viewer`` client given by :attr:`IsaacSimBackendCfg.app_client_root` or
  ``ISAACLAB_ISAAC_SIM_APP_CLIENT_ROOT``.

This package keeps imports lazy so configuration-only imports avoid pulling the
Isaac Sim 7 libraries before they are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .isaac_sim_visualizer_cfg import IsaacSimBackendCfg

if TYPE_CHECKING:
    from .isaac_sim_visualizer import IsaacSimVisualizer

__all__ = ["IsaacSimBackendCfg", "IsaacSimVisualizer"]


def __getattr__(name: str):
    if name == "IsaacSimVisualizer":
        from .isaac_sim_visualizer import IsaacSimVisualizer

        return IsaacSimVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
