# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A/B comparison of a robot asset against its Hugging Face counterpart.

Subcommands:

* ``fetch``: download one asset folder from a Hugging Face dataset at a pinned revision.
* ``fetch-lab``: mirror one folder of the current Isaac Lab asset bucket (arm A) for local inspection.
* ``inspect``: dump the articulation structure (links, joints, mass, colliders) of a USD file.
* ``diff``: compare two structure dumps and report renamed, missing, and changed links and joints.
* ``run``: train arm ``A`` (the task's current asset) and arm ``B`` (a replacement USD) with the
  training benchmark for several seeds, writing one benchmark bundle per run.
* ``report``: aggregate the bundles of a ``run`` into a head-to-head KPI table.

Example::

    uv run python tools/asset_ab/asset_ab.py fetch frankarobotics_frankapanda_franka --out /tmp/ab/hf
    uv run python tools/asset_ab/asset_ab.py inspect <usd> --out /tmp/ab/b.json
    uv run python tools/asset_ab/asset_ab.py diff /tmp/ab/a.json /tmp/ab/b.json
    uv run python tools/asset_ab/asset_ab.py run --task Isaac-Reach-Franka --b-usd <usd> --out /tmp/ab/reach
    uv run python tools/asset_ab/asset_ab.py report /tmp/ab/reach
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "nvidia/isaacsim-assets"
DEFAULT_ROBOT_PATH = "env.scene.robot.spawn.usd_path"

# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


def cmd_fetch(args: argparse.Namespace) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    # The repo-wide file listing is truncated on large datasets, so list the asset folder explicitly.
    api = HfApi()
    files = [
        entry.path
        for entry in api.list_repo_tree(
            args.dataset, path_in_repo=args.folder, recursive=True, repo_type="dataset", revision=args.revision
        )
        if hasattr(entry, "size")
    ]
    if not files:
        print(f"No files found under {args.folder!r} in {args.dataset}", file=sys.stderr)
        return 1
    for path in files:
        hf_hub_download(args.dataset, path, repo_type="dataset", revision=args.revision, local_dir=args.out)
    print(f"Downloaded {len(files)} files from {args.dataset}@{args.revision or 'main'} to {args.out}/{args.folder}")
    for path in files:
        if path.endswith((".usd", ".usda", ".usdc")) and "/payloads/" not in path and "/dependencies/" not in path:
            print(f"  {path}")
    return 0


def cmd_fetch_lab(args: argparse.Namespace) -> int:
    import urllib.parse
    import urllib.request
    import xml.etree.ElementTree as ET

    from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR

    # NUCLEUS_ASSET_ROOT_DIR is "<bucket url>/Assets" for the S3 layout, so strip the key prefix to get the bucket.
    parsed = urllib.parse.urlparse(NUCLEUS_ASSET_ROOT_DIR)
    bucket = f"{parsed.scheme}://{parsed.netloc}"
    prefix = f"{parsed.path.strip('/')}/Isaac/{args.folder.strip('/')}/"
    keys, token = [], None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        with urllib.request.urlopen(f"{bucket}/?{urllib.parse.urlencode(query)}") as response:
            root = ET.fromstring(response.read())
        ns = {"s": root.tag.split("}")[0].strip("{")}
        keys += [k.text for k in root.findall(".//s:Contents/s:Key", ns) if "/.thumbs/" not in k.text]
        token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not token:
            break
    for key in keys:
        dest = Path(args.out) / key[len(prefix) :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            urllib.request.urlretrieve(f"{bucket}/{urllib.parse.quote(key)}", dest)
    print(f"Mirrored {len(keys)} files from {bucket}/{prefix} to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def inspect_usd(path: str, variants: dict[str, str] | None = None) -> dict:
    """Return a JSON-serializable summary of the articulation structure of ``path``."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(path)
    if stage is None:
        raise FileNotFoundError(path)
    default_prim = stage.GetDefaultPrim()
    available_variants = {}
    if default_prim:
        variant_sets = default_prim.GetVariantSets()
        for name in variant_sets.GetNames():
            variant_set = variant_sets.GetVariantSet(name)
            if variants and name in variants:
                variant_set.SetVariantSelection(variants[name])
            available_variants[name] = {
                "selected": variant_set.GetVariantSelection(),
                "choices": variant_set.GetVariantNames(),
            }

    links, joints, articulation_roots = [], [], []
    collider_count = disabled_colliders = mesh_count = instanceable = 0
    # Instance proxies must be traversed, otherwise colliders and meshes inside instanceable assets are missed.
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsInstanceable():
            instanceable += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_count += 1
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is False:
                disabled_colliders += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
            links.append({"name": prim.GetName(), "path": str(prim.GetPath()), "mass": mass})
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            body1 = [str(p) for p in joint.GetBody1Rel().GetTargets()]
            name = prim.GetName()
            if name.startswith("Physics") and body1:
                # Generic names (PhysicsFixedJoint) repeat; qualify them by the child link.
                name = f"{name}->{body1[0].rsplit('/', 1)[-1]}"
            entry = {
                "name": name,
                "type": prim.GetTypeName(),
                "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                "body1": body1,
            }
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                typed = (
                    UsdPhysics.RevoluteJoint(prim)
                    if prim.IsA(UsdPhysics.RevoluteJoint)
                    else UsdPhysics.PrismaticJoint(prim)
                )
                entry["lower"] = typed.GetLowerLimitAttr().Get()
                entry["upper"] = typed.GetUpperLimitAttr().Get()
            joints.append(entry)

    masses = [link["mass"] for link in links if link["mass"] is not None]
    return {
        "path": path,
        "variants": available_variants,
        "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
        "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "articulation_roots": articulation_roots,
        "links": links,
        "joints": joints,
        "total_mass": sum(masses) if masses else None,
        "links_without_mass": len(links) - len(masses),
        "collider_count": collider_count,
        "disabled_collider_count": disabled_colliders,
        "mesh_count": mesh_count,
        "instanceable_prims": instanceable,
    }


def cmd_wrap(args: argparse.Namespace) -> int:
    """Author a USD that references usd and pre-selects variants, so no task config change is needed."""
    from pxr import Usd

    source = Usd.Stage.Open(args.usd)
    default_prim = source.GetDefaultPrim()
    stage = Usd.Stage.CreateNew(args.out)
    prim = stage.DefinePrim(f"/{default_prim.GetName()}")
    prim.GetReferences().AddReference(str(Path(args.usd).resolve()))
    stage.SetDefaultPrim(prim)
    for item in args.variant:
        name, value = item.split("=", 1)
        prim.GetVariantSets().GetVariantSet(name).SetVariantSelection(value)
    stage.GetRootLayer().Save()
    print(f"Wrote {args.out} ({len(args.variant)} variant selections)")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    summary = inspect_usd(args.usd, dict(item.split("=", 1) for item in args.variant or []))
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(
        f"{args.usd}\n  links={len(summary['links'])} joints={len(summary['joints'])} "
        f"variants={summary['variants']}\n  mass={summary['total_mass']} colliders={summary['collider_count']} meshes={summary['mesh_count']}"
    )
    return 0


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def diff_structures(a: dict, b: dict) -> dict:
    """Compare link and joint names between two structure dumps."""
    result: dict = {}
    for kind in ("links", "joints"):
        names_a = [item["name"] for item in a[kind]]
        names_b = [item["name"] for item in b[kind]]
        only_a = [n for n in names_a if n not in names_b]
        only_b = [n for n in names_b if n not in names_a]
        # Pair leftovers that differ only by case or punctuation (e.g. panda_link0 vs panda_Link_0).
        by_norm_b = {_norm(n): n for n in only_b}
        renamed = {n: by_norm_b[_norm(n)] for n in only_a if _norm(n) in by_norm_b}
        result[kind] = {
            "count_a": len(names_a),
            "count_b": len(names_b),
            "only_in_a": [n for n in only_a if n not in renamed],
            "only_in_b": [n for n in only_b if n not in renamed.values()],
            "probable_renames": renamed,
            "same_order": [n for n in names_a if n in names_b] == [n for n in names_b if n in names_a],
        }
    limits = {}
    joints_b = {j["name"]: j for j in b["joints"]}
    for joint in a["joints"]:
        other = joints_b.get(joint["name"])
        if other and (joint.get("lower"), joint.get("upper")) != (other.get("lower"), other.get("upper")):
            limits[joint["name"]] = {
                "a": (joint.get("lower"), joint.get("upper")),
                "b": (other.get("lower"), other.get("upper")),
            }
    result["joint_limit_changes"] = limits
    for key in ("total_mass", "collider_count", "mesh_count", "meters_per_unit", "up_axis", "instanceable_prims"):
        result[key] = {"a": a[key], "b": b[key], "equal": a[key] == b[key]}
    result["compatible_names"] = not (
        result["links"]["only_in_a"]
        or result["joints"]["only_in_a"]
        or result["links"]["probable_renames"]
        or result["joints"]["probable_renames"]
    )
    return result


def cmd_diff(args: argparse.Namespace) -> int:
    result = diff_structures(json.loads(Path(args.a).read_text()), json.loads(Path(args.b).read_text()))
    print(json.dumps(result, indent=2))
    if not result["compatible_names"]:
        print("\nNames differ: arm B needs task config overrides (pass them with `run --b-override`).", file=sys.stderr)
    return 0 if result["compatible_names"] else 1


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def build_command(args: argparse.Namespace, arm: str, seed: int, out_dir: Path) -> list[str]:
    cmd = [
        "uv", "run", "python", "scripts/benchmarks/training.py",
        "--rl_library", args.rl_library,
        "--task", args.task,
        "--max_iterations", str(args.max_iterations),
        "--seed", str(seed),
        "--output_path", str(out_dir),
    ]  # fmt: skip
    if args.num_envs:
        cmd += ["--num_envs", str(args.num_envs)]
    cmd += list(args.extra or [])
    if arm == "B":
        cmd.append(f"{args.robot_usd_key}={args.b_usd}")
        cmd += list(args.b_override or [])
    else:
        cmd += list(args.a_override or [])
    return cmd


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    for seed in args.seeds:
        for arm in args.arms:
            run_dir = out / f"{arm}_seed{seed}"
            if list(run_dir.glob("benchmark_training*.json")) and not args.force:
                print(f"skip {run_dir} (bundle exists; use --force to rerun)")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_command(args, arm, seed, run_dir)
            (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")
            print(f"[{arm} seed={seed}] {' '.join(cmd)}", flush=True)
            if args.dry_run:
                continue
            with open(run_dir / "train.log", "w") as log:
                code = subprocess.call(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            print(f"  exit={code} log={run_dir / 'train.log'}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def _get(bundle: dict, *keys, default=None):
    node = bundle
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def read_kpis(bundle: dict) -> dict[str, float | None]:
    """Extract the head-to-head KPIs from one benchmark bundle."""
    return {
        "reward_final_ema": _get(bundle, "learning", "reward", "final_ema"),
        "ep_length_final_ema": _get(bundle, "learning", "ep_length", "final_ema"),
        "success_rate": bundle.get("success_rate"),
        "total_fps": _get(bundle, "runtime", "total_fps", "mean"),
        "collection_fps": _get(bundle, "runtime", "collection_fps", "mean"),
        "iteration_time_s": _get(bundle, "runtime", "iteration_time_s", "mean"),
        "wall_time_s": _get(bundle, "runtime", "total_wall_time_s"),
        "gpu_mem_gb_peak": _get(bundle, "resources", "gpu_mem_gb", "peak"),
        "gpu_util_pct": _get(bundle, "resources", "gpu_util_pct", "mean"),
        "ram_gb_peak": _get(bundle, "resources", "ram_gb", "peak"),
    }


def collect(root: Path) -> dict[str, list[dict]]:
    arms: dict[str, list[dict]] = {}
    for run_dir in sorted(root.glob("*_seed*")):
        files = sorted(run_dir.glob("benchmark_training*.json"))
        if not files:
            continue
        arm = run_dir.name.split("_seed")[0]
        arms.setdefault(arm, []).append(read_kpis(json.loads(files[-1].read_text())))
    return arms


def _stats(values: list) -> tuple[float, float] | None:
    values = [v for v in values if v is not None]
    if not values:
        return None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def format_report(arms: dict[str, list[dict]]) -> str:
    names = sorted(arms)
    kpis = list(next(iter(arms.values()))[0]) if arms else []
    lines = [
        "| KPI | " + " | ".join(f"{n} (n={len(arms[n])})" for n in names) + " | B vs A |",
        "|---|" + "---|" * (len(names) + 1),
    ]
    for kpi in kpis:
        cells, means = [], {}
        for name in names:
            stats = _stats([run[kpi] for run in arms[name]])
            means[name] = stats[0] if stats else None
            cells.append(f"{stats[0]:.4g} ± {stats[1]:.2g}" if stats else "n/a")
        delta = "n/a"
        if means.get("A") not in (None, 0) and means.get("B") is not None:
            delta = f"{100 * (means['B'] - means['A']) / abs(means['A']):+.1f}%"
        lines.append(f"| {kpi} | " + " | ".join(cells) + f" | {delta} |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    arms = collect(Path(args.root))
    if not arms:
        print(f"No benchmark bundles found under {args.root}", file=sys.stderr)
        return 1
    report = format_report(arms)
    print(report)
    if args.out:
        Path(args.out).write_text(report + "\n")
    return 0


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="Download an asset folder from a Hugging Face dataset.")
    p.add_argument("folder")
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--revision", default=None, help="Commit hash or tag to pin.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("fetch-lab", help="Mirror a folder of the current Isaac Lab asset bucket.")
    p.add_argument("folder", help="Folder below the Isaac root, e.g. IsaacLab/Robots/FrankaEmika.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_fetch_lab)

    p = sub.add_parser("wrap", help="Author a USD that references an asset with variants pre-selected.")
    p.add_argument("usd")
    p.add_argument("--variant", action="append", required=True, metavar="SET=VALUE")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_wrap)

    p = sub.add_parser("inspect", help="Dump the articulation structure of a USD file.")
    p.add_argument("usd")
    p.add_argument("--variant", action="append", metavar="SET=VALUE", help="Select a variant before inspecting.")
    p.add_argument("--out")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("diff", help="Compare two structure dumps.")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("run", help="Train arms A and B for several seeds.")
    p.add_argument("--task", required=True)
    p.add_argument("--b-usd", required=True, help="Local path of the replacement USD used by arm B.")
    p.add_argument("--robot-usd-key", default=DEFAULT_ROBOT_PATH, help="Hydra key of the robot USD path.")
    p.add_argument("--rl-library", default="rsl_rl")
    p.add_argument("--max-iterations", type=int, default=1000)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--arms", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--a-override", nargs="*", help="Extra Hydra overrides for arm A only.")
    p.add_argument("--b-override", nargs="*", help="Extra Hydra overrides for arm B only (renamed links or joints).")
    p.add_argument("--extra", nargs="*", help="Extra arguments passed to both arms.")
    p.add_argument("--out", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report", help="Aggregate the bundles of a run into a KPI table.")
    p.add_argument("root")
    p.add_argument("--out")
    p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
