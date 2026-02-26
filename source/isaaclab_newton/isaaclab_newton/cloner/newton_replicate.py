# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import torch

from pxr import Usd

from .runtime_clone_plan import NewtonClonePlan, register_plan

logger = logging.getLogger(__name__)


def newton_replicate(
    stage: Usd.Stage,
    sources: list[str],
    destinations: list[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims for Newton physics backend.

    Newton does not require explicit physics replication like PhysX. Instead,
    this captures the clone plan at runtime so downstream consumers can attempt
    optimized Newton model construction using scene-level clone metadata.

    Args:
        stage: USD stage.
        sources: Source prim paths.
        destinations: Destination templates with ``"{}"`` for env index.
        env_ids: Environment indices.
        mapping: Bool mask selecting envs per source.
        device: Unused for direct replication (kept for API compatibility).

    Returns:
        None
    """
    # Capture clone metadata for scene data providers.
    plan = NewtonClonePlan(
        sources=list(sources),
        destinations=list(destinations),
        env_ids=env_ids.detach().clone(),
        mapping=mapping.detach().clone(),
        positions=positions.detach().clone() if positions is not None else None,
        quaternions=quaternions.detach().clone() if quaternions is not None else None,
        up_axis=up_axis,
    )
    register_plan(stage, plan)
    logger.debug(
        "[newton_replicate] Recorded runtime clone plan: sources=%d envs=%d map_shape=%s",
        len(sources),
        int(env_ids.shape[0]) if hasattr(env_ids, "shape") else len(env_ids),
        tuple(mapping.shape) if hasattr(mapping, "shape") else "unknown",
    )
