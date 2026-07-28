---
name: isaaclab-regenerating-golden-images
description: Regenerates visualizer and renderer golden images when a feature or rendering quality change intentionally alters rendered output. Use when a golden test fails because the expected visual has changed — not when a bug introduced a regression (fix the bug instead of updating the golden).
audience: developer
status: stable
owners:
  - isaaclab-maintainers
---

# Regenerating Golden Images

## When To Use

Use this skill when a feature or quality change **intentionally** changes what the renderer or visualizer produces, making the existing golden image stale.

**Do not use this skill** when a golden test fails due to a bug. Fix the bug so the render matches the golden rather than updating the golden to match the broken output.

The two golden test suites are:

- **Visualizer goldens** — `source/isaaclab_visualizers/test/golden_images/<scene>/<backend>-<visualizer>-<mode>.png`
  Run via: `uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_newton.py`
  and: `uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_physx.py`

- **Renderer goldens** — `source/isaaclab_tasks/test/golden_images/<scene>/<backend>-<renderer>-<data_type>.png`
  Run via: `uv run python -m pytest source/isaaclab_tasks/test/test_<scene>_rendering.py`

## Workflow

### 1. Identify which goldens need regeneration

Run the affected test file with `--no-header -v` to get per-combination pass/fail output:

```bash
uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_newton.py -v --no-header
```

Comparison images (actual vs golden) are written to `tests/comparison-images/` when a check fails. Inspect the pair visually before proceeding.

Ask: is the difference the expected result of the feature change? If yes, proceed to regenerate. If the difference looks like a bug (wrong pose, missing geometry, corrupt output), stop and investigate.

### 2. Delete the stale golden file(s)

```bash
rm source/isaaclab_visualizers/test/golden_images/<scene>/<backend>-<visualizer>-<mode>.png
# or for renderer goldens:
rm source/isaaclab_tasks/test/golden_images/<scene>/<backend>-<renderer>-<type>.png
```

### 3. Run the test to save the new golden

On the first run after deletion the test auto-saves the current frame as the new golden and then fails with:

> `Golden image not found for <scene>/<filename>.png. Saved the current frame as the new golden at: <path>`

This is expected — the forced failure is a review gate.

### 4. Inspect the new golden

Open the newly saved PNG and verify:
- The scene content is correct (robot visible, cloth or deformable in a plausible pose).
- No obvious artifacts (all-black, all-white, missing geometry, wrong camera angle).
- The output matches what the feature change was expected to produce.

### 5. Run the test again to confirm it passes

```bash
uv run python -m pytest <path/to/test.py>::<test_id> -v --no-header -p no:flaky
```

The second run compares the new frame against the just-saved golden. It should pass within the configured thresholds.

If it fails with a threshold violation, consider:
- Whether the thresholds in `visualizer_golden_utils.py` (`_MAX_DIFF_PCT_OVERRIDES`, `_SSIM_THRESHOLD_OVERRIDES`) need updating for this combination.
- Whether the render is non-deterministic and the golden strategy needs adjustment (see Special Cases below).

### 6. Run the full golden suite to catch cross-test contamination

Because all combinations run in the same process (one `AppLauncher` per test file), a change to one combination can affect subsequent ones via GPU/RTX state accumulation. Run the full file before committing:

```bash
uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_newton.py -v --no-header
```

### 7. Commit the updated golden(s)

Include only the regenerated PNG(s) and any threshold changes. Add a changelog fragment for the affected package if the visual change is user-facing.

## Special Cases

### VBD cloth non-determinism (franka_cloth Newton modes)

VBD parallel reduction ordering is non-deterministic across CUDA invocations and GPUs: even after a single physics step the cloth position can vary by 30–60% between machines. Use **0 warmup steps** for Newton-visualizer franka_cloth combinations so the cloth is captured at reset (its initial USD-defined position), which is fully deterministic. The `newton-tiled` and `newton-viewport` modes for franka_cloth use this strategy in `visualizer_golden_utils.py`:

```python
n_warmup = 0 if visualizer_type == "newton" else _viz_utils._FRANKA_CLOTH_WARMUP_STEPS
```

Kit-visualizer modes still need `_FRANKA_CLOTH_WARMUP_STEPS = 1` step because the Kit viewport requires Newton FK to be propagated to USD Fabric before the arm links become visible.

### Kit RTX rendering variability

Kit's RTX renderer accumulates TAA (Temporal Anti-Aliasing) history and uses stochastic sampling. This makes Kit-mode goldens inherently noisier than Newton-GL or Newton-Warp goldens. Thresholds for `kit-*` mode combinations are therefore much looser (often 4–30%) than Newton-mode combinations (1–4%). When regenerating kit goldens, expect first-pass pixel diffs of 5–20% to still require retries in CI — the `FLAKY_MARK` (3 retries, 1 pass) handles this automatically.

### Cross-GPU variation (Newton GL viewport)

Newton GL rendering is deterministic on the same GPU but can show ~1–4% cross-GPU pixel variation (different RTX GPU families use slightly different rasterization paths). Goldens captured on one GPU (e.g. RTX PRO 4500) may need thresholds of 4–8% to tolerate CI runners that use a different GPU (e.g. L40S).

### Threshold lookup precedence

Thresholds are resolved in this order (most specific wins):

1. `<scene>-<visualizer>-<mode>` key in `_MAX_DIFF_PCT_OVERRIDES` / `_SSIM_THRESHOLD_OVERRIDES`
2. `<visualizer>-<mode>` key in the same dicts
3. `MAX_DIFF_PCT_BY_VISUALIZER[visualizer]` / `_SSIM_THRESHOLD_BY_VISUALIZER[visualizer]`

Update only the narrowest key that covers the failing combination.

## Validation

After regenerating, confirm:

```bash
# All franka_cloth combinations pass
uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_newton.py -k franka_cloth -v --no-header

# Full suite passes (no cross-test contamination)
uv run python -m pytest source/isaaclab_visualizers/test/test_visualizer_golden_newton.py -v --no-header

# Lint/format clean
uv run isaaclab -f
```

## Maintenance

Keep this skill synchronized with:
- `source/isaaclab_visualizers/test/visualizer_golden_utils.py` — threshold maps, warmup constants, `validate_visualizer_frame`.
- `source/isaaclab_tasks/test/rendering_test_utils.py` — renderer golden save/compare path.
- `AGENTS.md` — authoritative project conventions.

## References

- [Visualizer golden utils](../../../source/isaaclab_visualizers/test/visualizer_golden_utils.py)
- [Renderer rendering test utils](../../../source/isaaclab_tasks/test/rendering_test_utils.py)
- [Golden images — visualizer](../../../source/isaaclab_visualizers/test/golden_images/)
- [Golden images — renderer](../../../source/isaaclab_tasks/test/golden_images/)
