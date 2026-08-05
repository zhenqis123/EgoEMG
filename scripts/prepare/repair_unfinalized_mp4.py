"""Repair unfinalized MP4/MKV recordings missing the moov atom.

Some ShowEE wrist recordings were interrupted before the container was
finalized: the file contains ``ftyp + mdat`` only (mdat runs to EOF), so
ffprobe reports "moov atom not found".  The media data (length-prefixed
H.264 NAL units) is intact.

This script extracts the NAL stream from the mdat, converts it to
annex-b, and re-muxes it with ffmpeg so the recording becomes readable
again.  The repaired file is written next to the source as
``<name>_repaired.mp4``.

Usage::

    python scripts/prepare/repair_unfinalized_mp4.py \
        /path/to/recording.mkv [more files...]
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import sys
from pathlib import Path


def extract_annexb(mdat_path: Path, start: int, end: int,
                   out_path: Path) -> int:
    """Copy length-prefixed NAL units from mdat[start:end] to annex-b."""
    n_nals = 0
    with mdat_path.open("rb") as src, out_path.open("wb") as dst:
        src.seek(start)
        pos = start
        remaining = end - start
        while remaining >= 4:
            raw = src.read(4)
            (length,) = struct.unpack(">I", raw)
            pos += 4
            remaining -= 4
            if length == 0 or length > remaining:
                break  # padding or truncated tail
            payload = src.read(length)
            remaining -= length
            pos += length
            if payload:
                dst.write(b"\x00\x00\x00\x01")
                dst.write(payload)
                n_nals += 1
        else:
            pass
    return n_nals


def find_mdat(path: Path) -> tuple[int, int]:
    """Return (mdat_start, mdat_end) walking top-level atoms."""
    with path.open("rb") as f:
        off = 0
        while True:
            f.seek(off)
            hdr = f.read(8)
            if len(hdr) < 8:
                raise ValueError(f"truncated header at {off:#x}")
            size, typ = struct.unpack(">I4s", hdr)
            typ = typ.decode(errors="replace")
            if typ == "mdat":
                end = off + size if size > 0 else path.stat().st_size
                return off + 8, end
            if size == 0 or size < 8:
                raise ValueError(f"bad atom {typ} at {off:#x}")
            off += size


def repair(path: Path, keep_annexb: bool = False) -> dict:
    mdat_start, mdat_end = find_mdat(path)
    annexb = path.with_name(path.stem + ".annexb.h264")
    n_nals = extract_annexb(path, mdat_start, mdat_end, annexb)
    repaired = path.with_name(path.stem + "_repaired.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "h264", "-i", str(annexb),
           "-c", "copy", str(repaired)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"path": str(path), "status": "failed",
                "stderr": proc.stderr[-400:], "n_nals": n_nals}
    if not keep_annexb:
        annexb.unlink(missing_ok=True)
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,width,height",
         "-of", "csv=p=0", str(repaired)],
        capture_output=True, text=True)
    frames = r.stdout.strip() if r.returncode == 0 else "?"
    return {"path": str(path), "status": "ok", "n_nals": n_nals,
            "frames": frames, "repaired": str(repaired)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--keep-annexb", action="store_true")
    args = ap.parse_args()
    for path in args.files:
        print(repair(path, args.keep_annexb))


if __name__ == "__main__":
    main()
