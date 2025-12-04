#!/usr/bin/env python3
"""
Two-column PDF extractor for the Enenlhet New Testament.

- Splits each page into left/right columns using page geometry or auto-detected gutter.
- Optional: remove footnotes below a separating horizontal rule (with fallback heuristic).
- Removes headers/footers via adjustable margins.
- Writes a single concatenated .txt and optional per-page files.
- Optional: draw debug images with crop boxes and detected footnote separator.

Usage (recommended for your layout + footnote removal):
  python extract_enenlhet_nt.py --pdf "New_Testament_Enenlhet.pdf" \
      --out enenlhet_nt_all.txt --per_page_dir pages_txt \
      --auto_gutter --gap 3 --left_margin 6 --top_margin 6 --right_margin 24 --bottom_margin 54 \
      --x_tol 1.0 --y_tol 2.5 \
      --remove_footnotes --footnote_dir footnotes_txt \
      --debug_pages 1 --debug_dir debug_boxes

Notes:
- Increase bottom margin if a footer leaks in; increase top margin if a header leaks in.
- If a subset of pages mis-split, try a small --gutter_shift (e.g., 8..20).
"""

from pathlib import Path
import argparse
import pdfplumber

try:
    from PIL import ImageDraw

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


# -------------------------
# Extraction helpers
# -------------------------
def extract_column_text(page, bbox, x_tol=1.5, y_tol=3.0):
    cropped = page.crop(bbox)
    text = cropped.extract_text(x_tolerance=x_tol, y_tolerance=y_tol) or ""
    return text.strip()


def autodetect_gutter_x(page, x0, x1, y0, y1, x_tol=1.0, y_tol=2.5):
    """
    Returns an x-coordinate for the gutter between two columns,
    inferred by clustering word midpoints (1D k-means).
    Falls back to geometric mid if insufficient words found.
    """
    region = page.crop((x0, y0, x1, y1))
    try:
        words = region.extract_words(
            x_tolerance=x_tol, y_tolerance=y_tol, keep_blank_chars=False
        )
    except Exception:
        words = None

    if not words or len(words) < 10:
        return (x0 + x1) / 2.0

    xmids = []
    for w in words:
        try:
            xmids.append((float(w["x0"]) + float(w["x1"])) / 2.0)
        except Exception:
            pass

    if len(xmids) < 10:
        return (x0 + x1) / 2.0

    mn, mx = min(xmids), max(xmids)
    c1, c2 = mn + (mx - mn) * 0.33, mn + (mx - mn) * 0.66
    for _ in range(20):
        left_grp = [x for x in xmids if abs(x - c1) <= abs(x - c2)]
        right_grp = [x for x in xmids if abs(x - c2) < abs(x - c1)]
        if not left_grp or not right_grp:
            break
        new_c1, new_c2 = sum(left_grp) / len(left_grp), sum(right_grp) / len(right_grp)
        if abs(new_c1 - c1) < 0.05 and abs(new_c2 - c2) < 0.05:
            c1, c2 = new_c1, new_c2
            break
        c1, c2 = new_c1, new_c2

    left_center, right_center = sorted([c1, c2])
    gutter = (left_center + right_center) / 2.0
    return max(x0 + 2, min(x1 - 2, gutter))


# -------------------------
# Footnote detection
# -------------------------
def find_horizontal_separator_y(
    page, x0, x1, y0, y1, min_width_ratio=0.6, search_lower_ratio=0.4, y_tolerance=1.0
):
    """
    Look for a horizontal rule separating main text from footnotes.

    Returns:
        y_sep (float) if found, else None.
    """
    # pdfplumber exposes line objects via page.lines in newer versions, or page.objects["line"]
    lines = getattr(page, "lines", None)
    if lines is None:
        lines = page.objects.get("line", [])

    if not lines:
        return None

    W = x1 - x0
    min_width = W * float(min_width_ratio)

    # Work in the lower portion of the working area to avoid header rules
    y_min = y0 + (y1 - y0) * float(search_lower_ratio)
    candidates = []
    for ln in lines:
        try:
            lx0, ly0, lx1, ly1 = (
                float(ln["x0"]),
                float(ln["y0"]),
                float(ln["x1"]),
                float(ln["y1"]),
            )
        except Exception:
            continue

        length = abs(lx1 - lx0)
        horizontal = abs(ly0 - ly1) <= y_tolerance
        spans_width = length >= min_width
        within_x = max(lx0, x0) <= min(lx1, x1)  # overlaps our working x-range
        within_y = (ly0 >= y_min) and (ly0 <= y1)

        if horizontal and spans_width and within_x and within_y:
            # Prefer the first strong rule above the bottom margin
            candidates.append((ly0, lx0, lx1))

    if not candidates:
        return None

    # Choose the highest such line nearest to the bottom (i.e., with the largest y)
    # NOTE: pdf coordinate systems can vary; this strategy works if y increases downward or upward,
    # because we restrict within the lower band and pick the one closest to the text area.
    candidates.sort(key=lambda t: t[0])  # sort by y
    # Heuristic: pick the median/upper-half to avoid page bottom border if present
    return candidates[-1][0]


def fallback_wide_text_y(
    page,
    x0,
    x1,
    y0,
    y1,
    mid,
    gap,
    x_tol=1.0,
    y_tol=2.5,
    lower_band_ratio=0.6,
    cross_gutter_px=10,
):
    """
    Fallback if no line is found: detect footnote by finding text that spans across both columns.
    Returns the top y of such wide text block (or None).
    """
    region = page.crop((x0, y0, x1, y1))
    words = (
        region.extract_words(
            x_tolerance=x_tol, y_tolerance=y_tol, keep_blank_chars=False
        )
        or []
    )
    if not words:
        return None

    y_min = y0 + (y1 - y0) * float(lower_band_ratio)
    threshold_left = mid - float(cross_gutter_px)
    threshold_right = mid + float(cross_gutter_px)

    # Look for any word box that crosses the gutter and sits in the lower band
    ys = []
    for w in words:
        try:
            wx0, wx1, wy0, wy1 = (
                float(w["x0"]),
                float(w["x1"]),
                float(w["top"]),
                float(w["bottom"]),
            )
        except Exception:
            # Some pdfplumber versions provide y0/y1 instead of top/bottom
            try:
                wy0, wy1 = float(w["y0"]), float(w["y1"])
                wx0, wx1 = float(w["x0"]), float(w["x1"])
            except Exception:
                continue

        crosses = (wx0 < threshold_left) and (wx1 > threshold_right)
        in_lower = (wy0 >= y_min) or (wy1 >= y_min)
        if crosses and in_lower:
            ys.append(min(wy0, wy1))

    if not ys:
        return None

    # top of the wide text cluster
    return min(ys)


# -------------------------
# Debug helpers
# -------------------------
def draw_debug_boxes(
    page, left_bbox, right_bbox, out_path, resolution=150, sep_y=None, x0=None, x1=None
):
    if not PIL_AVAILABLE:
        return
    im = page.to_image(resolution=resolution)
    im.draw_rects([left_bbox], stroke="red", stroke_width=3)
    im.draw_rects([right_bbox], stroke="blue", stroke_width=3)

    # Draw separator line if detected
    if sep_y is not None and x0 is not None and x1 is not None:
        im.draw_rects(
            [(x0, sep_y - 0.5, x1, sep_y + 0.5)], stroke="green", stroke_width=3
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(str(out_path))


def dump_crops(page, left_bbox, right_bbox, out_dir, pageno, resolution=200):
    if not PIL_AVAILABLE:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    page.crop(left_bbox).to_image(resolution=resolution).save(
        str(out_dir / f"page_{pageno:04d}_left.png")
    )
    page.crop(right_bbox).to_image(resolution=resolution).save(
        str(out_dir / f"page_{pageno:04d}_right.png")
    )


# -------------------------
# Main
# -------------------------
def parse_debug_pages(arg: str):
    pages = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a, b = int(a), int(b)
                pages.update(range(min(a, b), max(a, b) + 1))
            except ValueError:
                continue
        else:
            try:
                pages.add(int(part))
            except ValueError:
                continue
    return pages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, help="Path to the input PDF")
    ap.add_argument(
        "--out", default="enenlhet_nt_text.txt", help="Path for combined TXT output"
    )
    ap.add_argument(
        "--per_page_dir",
        default="",
        help="Optional directory to save per-page TXT files",
    )

    # Margins tuned for flush top-left layout
    ap.add_argument("--left_margin", type=float, default=6.0)
    ap.add_argument("--right_margin", type=float, default=24.0)
    ap.add_argument("--top_margin", type=float, default=6.0)
    ap.add_argument("--bottom_margin", type=float, default=54.0)

    # Gutter handling
    ap.add_argument(
        "--auto_gutter", action="store_true", help="Auto-detect gutter per page"
    )
    ap.add_argument(
        "--gutter_shift",
        type=float,
        default=0.0,
        help="Manual split shift (points; + shifts right)",
    )
    ap.add_argument(
        "--gap",
        type=float,
        default=3.0,
        help="Gap (points) to prevent bbox overlap across gutter",
    )

    # Text grouping tolerances
    ap.add_argument("--x_tol", type=float, default=1.0)
    ap.add_argument("--y_tol", type=float, default=2.5)

    # Footnotes
    ap.add_argument(
        "--remove_footnotes",
        action="store_true",
        help="Detect separator and remove footnotes area",
    )
    ap.add_argument(
        "--footnote_dir",
        default="",
        help="Optional dir to save extracted footnotes per page",
    )
    ap.add_argument(
        "--footnote_pad",
        type=float,
        default=20.0,
        help="Padding above separator before cropping",
    )
    ap.add_argument(
        "--sep_min_width_ratio",
        type=float,
        default=0.6,
        help="Min width ratio for separator line",
    )
    ap.add_argument(
        "--sep_search_lower_ratio",
        type=float,
        default=0.25,
        help="Start searching in lower fraction of page",
    )
    ap.add_argument(
        "--fallback_lower_ratio",
        type=float,
        default=0.5,
        help="Lower band start for wide-text fallback",
    )

    # Debug
    ap.add_argument(
        "--debug_pages", default="", help="Comma/range list, e.g. '1,2,10-12' (1-based)"
    )
    ap.add_argument(
        "--debug_dir", default="", help="Directory for debug images with crop boxes"
    )
    ap.add_argument("--debug_dpi", type=int, default=150)

    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)
    per_page_dir = Path(args.per_page_dir) if args.per_page_dir else None
    if per_page_dir:
        per_page_dir.mkdir(parents=True, exist_ok=True)

    footnote_dir = Path(args.footnote_dir) if args.footnote_dir else None
    if footnote_dir:
        footnote_dir.mkdir(parents=True, exist_ok=True)

    debug_pages = parse_debug_pages(args.debug_pages) if args.debug_pages else set()
    debug_dir = Path(args.debug_dir) if args.debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    with pdf_plumber_open(pdf_path) as pdf, out_path.open("w", encoding="utf-8") as out:
        for i, page in enumerate(pdf.pages, start=1):
            W, H = page.width, page.height

            # Working area after margins
            x0 = args.left_margin
            x1 = W - args.right_margin
            y0 = args.top_margin
            y1 = H - args.bottom_margin
            if x1 <= x0 or y1 <= y0:
                x0, y0, x1, y1 = 0, 0, W, H

            # Optional footnote separator detection
            sep_y = None
            if args.remove_footnotes:
                sep_y = find_horizontal_separator_y(
                    page,
                    x0,
                    x1,
                    y0,
                    y1,
                    min_width_ratio=args.sep_min_width_ratio,
                    search_lower_ratio=args.sep_search_lower_ratio,
                )
                if sep_y is None:
                    # Fallback: look for wide text spanning both columns in the lower band
                    if args.auto_gutter:
                        mid = autodetect_gutter_x(
                            page, x0, x1, y0, y1, args.x_tol, args.y_tol
                        )
                    else:
                        mid = (x0 + x1) / 2.0
                    mid += args.gutter_shift
                    sep_y = fallback_wide_text_y(
                        page,
                        x0,
                        x1,
                        y0,
                        y1,
                        mid,
                        args.gap,
                        x_tol=args.x_tol,
                        y_tol=args.y_tol,
                        lower_band_ratio=args.fallback_lower_ratio,
                    )

                # If we have a separator, adjust the bottom of our text area (y1)
                if sep_y is not None:
                    # Save footnotes if requested
                    if footnote_dir:
                        fn_text = (
                            page.crop((x0, sep_y, x1, y1)).extract_text(
                                x_tolerance=args.x_tol, y_tolerance=args.y_tol
                            )
                            or ""
                        )
                        (footnote_dir / f"page_{i:04d}_footnotes.txt").write_text(
                            fn_text.strip() + "\n", encoding="utf-8"
                        )
                    y1 = max(y0, sep_y - args.footnote_pad)

            # Determine gutter split
            if args.auto_gutter:
                mid = autodetect_gutter_x(page, x0, x1, y0, y1, args.x_tol, args.y_tol)
                mid += args.gutter_shift
            else:
                mid = (x0 + x1) / 2.0 + args.gutter_shift

            # Non-overlapping column boxes
            gap = max(0.0, args.gap)
            left_bbox = (x0, y0, max(x0, min(mid - gap, x1)), y1)
            right_bbox = (min(x1, max(mid + gap, x0)), y0, x1, y1)

            left_text = extract_column_text(page, left_bbox, args.x_tol, args.y_tol)
            right_text = extract_column_text(page, right_bbox, args.x_tol, args.y_tol)

            page_text = f"=== PAGE {i} ===\n{left_text}\n{right_text}\n"
            out.write(page_text + "\n")

            if per_page_dir:
                (per_page_dir / f"page_{i:04d}.txt").write_text(
                    f"{left_text}\n{right_text}\n", encoding="utf-8"
                )

            # Debug image with boxes + separator if requested
            if debug_dir and (not debug_pages or i in debug_pages):
                dbg_path = debug_dir / f"page_{i:04d}.png"
                draw_debug_boxes(
                    page,
                    left_bbox,
                    right_bbox,
                    dbg_path,
                    resolution=args.debug_dpi,
                    sep_y=sep_y,
                    x0=x0,
                    x1=x1,
                )
                dump_crops(
                    page, left_bbox, right_bbox, debug_dir / "crops", i, resolution=200
                )

    print(f"Done. Combined text → {out_path}")
    if per_page_dir:
        print(f"Per-page files → {per_page_dir}")
    if footnote_dir:
        print(f"Footnotes (per page) → {footnote_dir}")
    if debug_dir:
        print(f"Debug images → {debug_dir}")


# Context manager to keep pdfplumber open/close cleanly
class pdf_plumber_open:
    def __init__(self, path):
        self.path = path
        self._pdf = None

    def __enter__(self):
        self._pdf = pdfplumber.open(self.path)
        return self._pdf

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._pdf:
                self._pdf.close()
        finally:
            return False  # propagate exceptions


if __name__ == "__main__":
    main()
