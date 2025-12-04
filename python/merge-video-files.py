#!/usr/bin/env python3
"""
Concatenate mixed VOB/MP4/MOV chunks per Disk into MP4 (H.264 + AAC).

Expected pattern (case-insensitive):
  CLA-00171_01_Disk1.VOB
  CLA-00171_02_Disk1.MP4
  CLA-00171_03_Disk1.MOV
  ...
Run this script in the folder with the files:

    ./merge_video_per_disk.py

Outputs: ./merged_mp4/CLA-00171_Disk1.mp4, etc.
"""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path.cwd()
OUTDIR = ROOT / "merged_mp4"
OUTDIR.mkdir(exist_ok=True)

# e.g., CLA-00171_01_Disk1.VOB  (extension: vob|mp4|mov, any case)
PATTERN = re.compile(r"^(CLA-\d{5})_(\d{2})_Disk(\d+)\.(vob|mp4|mov)$", re.IGNORECASE)


def seq_key(p: Path) -> int:
    m = PATTERN.match(p.name)
    return int(m.group(2)) if m else 0


def concat_mixed_to_mp4(files, base_id: str, outpath: Path):
    """
    Build a filter_complex concat graph that decodes each input and concatenates
    into a single H.264/AAC MP4. We append a deinterlacer (yadif) at the end
    (harmless on progressive sources, useful for VOB/DVD).
    """
    # Build input args: -i file1 -i file2 ...
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for f in files:
        cmd += ["-i", str(f)]

    n = len(files)
    # Build the concat filter: [0:v][0:a][1:v][1:a]... concat=n=N:v=1:a=1 [v][a]; [v]yadif[vf]
    parts = []
    for i in range(n):
        parts.append(f"[{i}:v]")
        parts.append(f"[{i}:a]")
    filtergraph = "".join(parts) + f"concat=n={n}:v=1:a=1[v][a];[v]yadif[vf]"

    cmd += [
        "-filter_complex",
        filtergraph,
        "-map",
        "[vf]",
        "-map",
        "[a]",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(outpath),
    ]

    print(f"→ Disk {disk_no}: merging {n} files → {outpath.name}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # If some clips lack audio, fall back to video-only concat
        print(f"   ! Audio track issue detected, retrying without audio …")
        parts = [f"[{i}:v]" for i in range(n)]
        filtergraph_vo = "".join(parts) + f"concat=n={n}:v=1:a=0[v];[v]yadif[vf]"
        cmd_vo = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
        for f in files:
            cmd_vo += ["-i", str(f)]
        cmd_vo += [
            "-filter_complex",
            filtergraph_vo,
            "-map",
            "[vf]",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(outpath),
        ]
        subprocess.run(cmd_vo, check=True)


# Collect target files (non-recursive; change to rglob if needed)
candidates = []
for ext in ("*.VOB", "*.vob", "*.MP4", "*.mp4", "*.MOV", "*.mov"):
    candidates += list(ROOT.glob(ext))

if not candidates:
    sys.exit("No VOB/MP4/MOV files found in this directory.")

# Group by (base_id, disk_no)
groups = {}
for p in candidates:
    m = PATTERN.match(p.name)
    if not m:
        continue
    base_id, seq, disk, ext = m.groups()
    groups.setdefault((base_id, disk), []).append(p)

if not groups:
    sys.exit("No filenames matched pattern CLA-xxxxx_nn_DiskN.(vob|mp4|mov)")

# Process each disk group
for (base_id, disk), files in sorted(
    groups.items(), key=lambda k: (k[0][0], int(k[0][1]))
):
    files.sort(key=seq_key)
    out = OUTDIR / f"{base_id}_Disk{disk}.mp4"
    concat_mixed_to_mp4(files, base_id, disk, out)

print(f"\nAll done. MP4s saved to: {OUTDIR.resolve()}")
