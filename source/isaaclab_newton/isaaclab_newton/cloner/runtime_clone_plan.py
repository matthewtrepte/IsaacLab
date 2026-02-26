"""Runtime clone plan registry for Newton scene build hints.

This module stores clone-plan metadata keyed by USD stage id so downstream
consumers (e.g., scene data providers) can attempt optimized Newton model
construction without requiring persistent USD metadata.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pxr import Usd, UsdUtils


@dataclass
class NewtonClonePlan:
    """In-memory clone plan captured during scene cloning."""

    sources: list[str]
    destinations: list[str]
    env_ids: torch.Tensor
    mapping: torch.Tensor
    positions: torch.Tensor | None = None
    quaternions: torch.Tensor | None = None
    up_axis: str = "Z"


_STAGE_PLAN_REGISTRY: dict[int, NewtonClonePlan] = {}


def _stage_id(stage: Usd.Stage) -> int:
    """Return stable stage id from USD stage cache."""
    return UsdUtils.StageCache.Get().Insert(stage).ToLongInt()


def register_plan(stage: Usd.Stage, plan: NewtonClonePlan) -> None:
    """Register or replace clone plan for stage."""
    _STAGE_PLAN_REGISTRY[_stage_id(stage)] = plan


def get_plan(stage: Usd.Stage) -> NewtonClonePlan | None:
    """Return clone plan for stage if available."""
    return _STAGE_PLAN_REGISTRY.get(_stage_id(stage))


def clear_plan(stage: Usd.Stage) -> None:
    """Remove clone plan for stage if present."""
    _STAGE_PLAN_REGISTRY.pop(_stage_id(stage), None)
