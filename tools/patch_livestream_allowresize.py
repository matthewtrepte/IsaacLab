#!/usr/bin/env python3
"""
Patch IsaacLab AppLauncher to add --/app/livestream/allowResize=true for
LIVESTREAM=1 and LIVESTREAM=2 modes (NVBug 6281418).

This patch is a standalone verification tool for QA.  It adds the missing
app-level resize flag that prevents NVST_R_BUSY when the OS resizes the
application window after a WebRTC client connects.  The setting mirrors
isaacsim.exp.full.streaming.kit which NVIDIA ships for this exact purpose.

Requirements:
  - IsaacLab with PR #7329 already applied (release/3.0.0 TOT or develop TOT).
    PR #7329 adds signalPort/streamPort/streamType/allowDynamicResize; this
    patch adds the companion app-level allowResize flag.

Usage (run from the IsaacLab root directory):
  python tools\\patch_livestream_allowresize.py

To undo:
  python tools\\patch_livestream_allowresize.py --undo
"""

import argparse
import shutil
import sys
from pathlib import Path

LAUNCHER = Path("source/isaaclab/isaaclab/app/app_launcher.py")
MARKER = '"--/app/livestream/allowResize=true",'

# The arg that must already be present (PR #7329 prerequisite check)
PREREQ = '"--/exts/omni.kit.livestream.app/primaryStream/streamType=webrtc",'

# Replace the streamType arg with streamType + allowResize, for both LIVESTREAM blocks.
# The replacement inserts the allowResize line with the same indentation as its siblings.
NEEDLE = '"--/exts/omni.kit.livestream.app/primaryStream/streamType=webrtc",'


def _detect_newline(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def apply(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    nl = _detect_newline(text)
    indent = "                    "  # 5 levels x 4 spaces

    if MARKER in text:
        print(f"Already applied — {path} is already patched. Nothing to do.")
        return

    if PREREQ not in text:
        sys.exit(
            f"ERROR: prerequisite pattern not found in {path}.\n"
            "This patch requires PR #7329 (LIVESTREAM=2 port/streamType settings)\n"
            "to already be present.  Make sure you are on release/3.0.0 TOT or\n"
            "develop TOT, then re-run this script."
        )

    replacement = f'{NEEDLE}{nl}{indent}{MARKER}'
    count_before = text.count(NEEDLE)
    if count_before != 2:
        sys.exit(
            f"ERROR: expected exactly 2 occurrences of the streamType arg "
            f"(one per LIVESTREAM block), found {count_before}. "
            "The file may have an unexpected structure."
        )

    patched = text.replace(NEEDLE, replacement)
    if patched.count(MARKER) != 2:
        sys.exit("ERROR: replacement produced an unexpected result. Aborting.")

    backup = path.with_suffix(path.suffix + ".orig")
    shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")

    print(f"Patched  : {path}")
    print(f"Backup   : {backup}")
    print()
    print("Setting added to LIVESTREAM=1 and LIVESTREAM=2 blocks:")
    print(f"  {MARKER}")
    print()
    print("Repro command (run from IsaacLab root):")
    print("  set LIVESTREAM=2")
    print("  set ENABLE_CAMERAS=1")
    print(r"  isaaclab.bat -p scripts\demos\bin_packing.py")
    print()
    print("Expected: NO 'NVST_R_BUSY' or 'NVST_R_INVALID_OPERATION' in the log.")


def undo(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".orig")
    if not backup.exists():
        sys.exit(f"No backup found at {backup}. Cannot undo.")
    shutil.copy2(backup, path)
    backup.unlink()
    print(f"Restored {path} from {backup}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--undo", action="store_true", help="Restore the original file from backup.")
    parser.add_argument("--file", default=str(LAUNCHER), help="Path to app_launcher.py (default: %(default)s)")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        sys.exit(
            f"Not found: {path}\n"
            "Run this script from the IsaacLab root directory, e.g.:\n"
            r"  python tools\patch_livestream_allowresize.py"
        )

    if args.undo:
        undo(path)
    else:
        apply(path)


if __name__ == "__main__":
    main()
