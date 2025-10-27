#!/usr/bin/env python3
"""
CLI: Transcribe audio (single file or batch directory) with Wav2Vec2,
write CSV(s), and optionally produce ELAN .eaf (new or merged).

Single-file:
  python asr2eaf.py \
    --wav path/to/audio.wav \
    --out-csv out/audio.csv \
    --segment-ms 20000 \
    --model sjhuskey/enenlhet-wav2vec2-model \
    --segment-dir work/segments \
    --to-eaf --new-eaf out/audio.eaf --tier-name ASR

Batch (process all WAVs in a dir, recursively):
  python asr2eaf.py \
    --batch-dir path/to/wavs \
    --recursive \
    --out-csv-template out/csv/{stem}.csv \
    --segment-dir-template work/segments/{stem} \
    --to-eaf \
    --new-eaf-template out/eaf/{stem}.eaf \
    --tier-name ASR
"""

import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterable
import itertools

import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from tqdm.auto import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import uuid

# ELAN
try:
    from pympi import Elan
except ImportError as e:
    raise SystemExit("Missing dependency: pip install pympi-ling") from e

TARGET_SR = 16000


# ------------------------------
# Transcription utilities
# ------------------------------
def segment_audio(input_wav: str, out_dir: str, segment_ms: int) -> List[Tuple[str, int, int, int]]:
    """Split WAV into fixed-length segments. Returns list of (seg_path, idx, start_ms, end_ms)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(input_wav)
    total_ms = len(audio)

    manifest = []
    idx = 0
    for start in range(0, total_ms, segment_ms):
        end = min(start + segment_ms, total_ms)
        seg = audio[start:end]
        seg_path = os.path.join(out_dir, f"segment_{idx:03}.wav")
        seg.export(seg_path, format="wav")
        manifest.append((seg_path, idx, start, end))
        idx += 1
    return manifest


def load_full_audio_mono_16k(
    wav_path: str, target_sr: int = TARGET_SR
) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(wav_path)  # [channels, time]
    wav = stereo_to_mono(wav)
    wav, sr = ensure_sr(wav, sr, target_sr)
    # shape: [1, time]
    return wav, sr


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


def stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def ensure_sr(waveform: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr

from typing import Iterable
import itertools


def resolve_media_from_eaf(eaf_path: str) -> Optional[str]:
    """Try to resolve a usable WAV path from EAF media descriptors."""

    eaf = Elan.Eaf(eaf_path)
    md = eaf.media_descriptors  # list of dicts
    if not md:
        return None
    eaf_dir = Path(eaf_path).parent.resolve()
    # Prefer RELATIVE_MEDIA_URL when present
    for m in md:
        rel = m.get("RELATIVE_MEDIA_URL")
        if rel:
            cand = (eaf_dir / rel).resolve()
            if cand.exists():
                return str(cand)
    # Fallback to absolute MEDIA_URL
    for m in md:
        absu = m.get("MEDIA_URL")
        if absu and Path(absu).exists():
            return absu
    return None


def eaf_windows(
    eaf_path: str, tier_names: Optional[Iterable[str]] = None
) -> List[Tuple[int, int]]:
    """Collect (start_ms, end_ms) windows from given EAF tiers (unique + sorted)."""

    eaf = Elan.Eaf(eaf_path)

    # Choose tiers
    if tier_names is None:
        # default: all time-alignable tiers
        tiers = [
            t
            for t in eaf.get_tier_names()
            if eaf.tiers[t]["ling_ref"] or eaf.tiers[t]["annotations"]
        ]
        # The above heuristic includes alignable tiers (time slots present). You can refine if needed.
    else:
        tiers = list(tier_names)

    spans = []
    for t in tiers:
        if t not in eaf.get_tier_names():
            continue
        # Returns list of tuples: (start, end, value)
        ann = eaf.get_annotation_data_for_tier(t)
        for s, e, _val in ann:
            if s is None or e is None:
                continue
            if e > s:
                spans.append((int(s), int(e)))

    # Deduplicate identical spans and sort
    spans = sorted(set(spans))
    return spans


@torch.no_grad()
def transcribe_window(
    full_wav: torch.Tensor,  # [1, time]
    sr: int,
    start_ms: int,
    end_ms: int,
    processor,
    model,
    device: str,
) -> str:
    # Convert ms to sample indices
    start_samp = int((start_ms / 1000.0) * sr)
    end_samp = int((end_ms / 1000.0) * sr)
    end_samp = max(end_samp, start_samp + 1)

    window = full_wav[:, start_samp:end_samp]  # [1, T]
    samples = window.squeeze(0).numpy()

    inputs = processor(
        samples, sampling_rate=sr, return_tensors="pt", return_attention_mask=True
    )
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)
    logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0].strip()

def pause_based_windows(
    wav_path: str,
    min_pause_ms: int = 150,
    silence_thresh: int | None = None,
    rel_thresh_db: int = 16,
    keep_silence_ms: int = 0,
    max_chunk_ms: int = 0,
) -> list[tuple[int, int]]:
    """
    Return [(start_ms, end_ms), ...] for continuous speech, where pauses >= min_pause_ms split chunks.
    silence_thresh: absolute dBFS (e.g., -40). If None, use audio.dBFS - rel_thresh_db.
    keep_silence_ms: pad each chunk on both ends (bounded by file edges).
    max_chunk_ms: further split long chunks (0 disables).
    """
    audio = AudioSegment.from_wav(wav_path)
    thresh = (
        silence_thresh if silence_thresh is not None else (audio.dBFS - rel_thresh_db)
    )

    spans = detect_nonsilent(
        audio,
        min_silence_len=min_pause_ms,
        silence_thresh=thresh,
        seek_step=1,  # ms resolution
    )
    # Apply padding and clamp
    total = len(audio)
    padded = []
    for s, e in spans:
        s = max(0, s - keep_silence_ms)
        e = min(total, e + keep_silence_ms)
        if e > s:
            padded.append((s, e))

    # Optionally split very long chunks
    if max_chunk_ms and max_chunk_ms > 0:
        split_spans = []
        for s, e in padded:
            cur = s
            while cur + max_chunk_ms < e:
                split_spans.append((cur, cur + max_chunk_ms))
                cur += max_chunk_ms
            if cur < e:
                split_spans.append((cur, e))
        return split_spans

    return padded


def transcribe_windows_to_rows(
    wav_path: str,
    windows: list[tuple[int, int]],
    processor,
    model,
    device,
) -> list[dict]:
    full_wav, sr = load_full_audio_mono_16k(wav_path, TARGET_SR)
    rows = []
    for idx, (s, e) in enumerate(tqdm(windows, desc=f"Transcribing pause-based")):
        text = transcribe_window(full_wav, sr, s, e, processor, model, device)
        rows.append(
            {
                "audio_file": os.path.basename(wav_path),
                "segment_index": idx,
                "start_ms": int(s),
                "end_ms": int(e),
                "start_s": f"{s/1000:.3f}",
                "end_s": f"{e/1000:.3f}",
                "segment_path": "",
                "text": text,
            }
        )
    return rows


@torch.no_grad()
def transcribe_file(seg_path: str, processor, model, device: str, target_sr: int = TARGET_SR) -> str:
    waveform, sr = torchaudio.load(seg_path)
    waveform = stereo_to_mono(waveform)
    waveform, sr = ensure_sr(waveform, sr, target_sr)
    samples = waveform.squeeze(0).numpy()

    inputs = processor(samples, sampling_rate=target_sr,
                       return_tensors="pt", return_attention_mask=True)
    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device)

    logits = model(input_values, attention_mask=attention_mask).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return text.strip()


def transcribe_to_csv(
    wav_path: str,
    out_csv: str,
    processor,
    model,
    device: str,
    segment_dir: str,
    segment_ms: int,
) -> List[Dict]:
    manifest = segment_audio(wav_path, segment_dir, segment_ms)
    manifest.sort(key=lambda x: x[1])

    rows = []
    for seg_path, idx, start_ms, end_ms in tqdm(manifest, desc=f"Transcribing {Path(wav_path).name}"):
        text = transcribe_file(seg_path, processor, model, device, TARGET_SR)
        rows.append({
            "audio_file": os.path.basename(wav_path),
            "segment_index": idx,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_s": f"{start_ms/1000:.3f}",
            "end_s": f"{end_ms/1000:.3f}",
            "segment_path": seg_path,
            "text": text
        })

    fieldnames = ["audio_file", "segment_index", "start_ms", "end_ms",
                  "start_s", "end_s", "segment_path", "text"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_csv} with {len(rows)} rows.")
    return rows


def read_segments_csv(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "audio_file": r["audio_file"],
                "segment_index": int(float(r["segment_index"])),
                "start_ms": int(float(r["start_ms"])),
                "end_ms": int(float(r["end_ms"])),
                "start_s": r.get("start_s", ""),
                "end_s": r.get("end_s", ""),
                "segment_path": r.get("segment_path", ""),
                "text": r["text"],
            })
    return rows


def transcribe_eaf_spans_to_rows(
    wav_path: str,
    eaf_path: str,
    processor,
    model,
    device: str,
    tiers: Optional[List[str]],
    min_ms: int,
) -> List[Dict]:
    full_wav, sr = load_full_audio_mono_16k(wav_path, TARGET_SR)
    tier_list = (
        None
        if tiers is None
        else (
            [t.strip() for t in ",".join(tiers).split(",")]
            if isinstance(tiers, list)
            else [t.strip() for t in tiers.split(",")]
        )
    )
    spans = eaf_windows(eaf_path, tier_list)

    rows = []
    for idx, (s, e) in enumerate(tqdm(spans, desc=f"Transcribing by EAF spans")):
        if (e - s) < min_ms:
            continue
        text = transcribe_window(full_wav, sr, s, e, processor, model, device)
        rows.append(
            {
                "audio_file": os.path.basename(wav_path),
                "segment_index": idx,
                "start_ms": int(s),
                "end_ms": int(e),
                "start_s": f"{s/1000:.3f}",
                "end_s": f"{e/1000:.3f}",
                "segment_path": "",  # not used in align-to-EAF mode
                "text": text,
            }
        )
    return rows


# ------------------------------
# EAF utilities (pympi-ling)
# ------------------------------
def build_new_eaf(
    segments,
    wav_path,
    out_eaf_path,
    tier_name="ASR",
    ling_type="default-lt",
    participant=None,
    relative_media=True,
    urn: str = None,
    generate_urn: bool = True,
    urn_namespace: str = "urn:enenlhet-asr-elan-eaf",
):

    eaf = Elan.Eaf()
    # --- URN PROPERTY ---
    if urn:
        add_urn_property(eaf, urn)
    elif generate_urn:
        add_urn_property(eaf, generate_eaf_urn(urn_namespace))

    # --- MEDIA DESCRIPTOR ---
    if wav_path:
        media_abs = str(Path(wav_path).resolve())
        rel = None
        if relative_media:
            # RELATIVE_MEDIA_URL should be relative to the EAF file location
            rel = os.path.relpath(
                media_abs, start=str(Path(out_eaf_path).parent.resolve())
            )
        # Pass absolute path, plus relpath for RELATIVE_MEDIA_URL
        eaf.add_linked_file(media_abs, relpath=rel)

    # --- LINGUISTIC TYPE (note the kw name: timealignable, no underscore) ---
    if ling_type not in eaf.linguistic_types:
        eaf.add_linguistic_type(ling_type, timealignable=True)

    # --- TIER ---
    eaf.add_tier(tier_name, ling=ling_type, part=participant if participant else None)

    # --- ANNOTATIONS ---
    for seg in segments:
        eaf.add_annotation(tier_name, seg["start_ms"], seg["end_ms"], seg["text"])

    eaf.add_property("ANNOTATOR", "ASR (Wav2Vec2 Enenlhet)")
    eaf.add_property("AUTHOR", "Generated automatically")

    Path(out_eaf_path).parent.mkdir(parents=True, exist_ok=True)
    eaf.to_file(out_eaf_path)
    print(f"Wrote new EAF -> {out_eaf_path}")


def merge_into_existing_eaf(
    existing_eaf_path: str,
    segments: List[Dict],
    out_eaf_path: str,
    tier_name: str = "ASR",
    ling_type: Optional[str] = None,
    participant: Optional[str] = None,
    ensure_urn_on_merge: bool = False,
    urn: Optional[str] = None,
    urn_namespace: str = "urn:enenlhet-asr-elan-eaf",
):
    eaf = Elan.Eaf(existing_eaf_path)

    if tier_name in eaf.get_tier_names():
        raise SystemExit(
            f"Tier '{tier_name}' already exists in {existing_eaf_path}. "
            "Choose a new --tier-name or remove it first."
        )

    chosen_lt = ling_type or next(iter(eaf.linguistic_types.keys()), "default-lt")
    if chosen_lt not in eaf.linguistic_types:
        eaf.add_linguistic_type(chosen_lt, timealignable=True)

    eaf.add_tier(tier_name, ling=chosen_lt, part=participant if participant else None)

    for seg in segments:
        eaf.add_annotation(tier_name, seg["start_ms"], seg["end_ms"], seg["text"])

    if ensure_urn_on_merge and not has_property(eaf, "URN"):
        add_urn_property(eaf, urn or generate_eaf_urn(urn_namespace))

    Path(out_eaf_path).parent.mkdir(parents=True, exist_ok=True)
    eaf.to_file(out_eaf_path)
    print(f"Merged ASR tier -> {out_eaf_path}")



# ------------------------------
# Helpers
# ------------------------------
def expand_template(template: str, wav_path: str) -> str:
    """Expand {stem} template using the input WAV path."""
    stem = Path(wav_path).stem
    return template.format(stem=stem)


def find_wavs(batch_dir: str, recursive: bool) -> List[str]:
    p = Path(batch_dir)
    pattern = "**/*.wav" if recursive else "*.wav"
    return [str(fp) for fp in p.glob(pattern) if fp.is_file()]


def generate_eaf_urn(namespace: str = "urn:nl-mpi-tools-elan-eaf") -> str:
    return f"{namespace}:{uuid.uuid4()}"


def has_property(eaf, name: str) -> bool:
    # pympi stores props in eaf.properties as list[tuple(name, value)]
    try:
        for k, v in getattr(eaf, "properties", []):
            if k == name:
                return True
    except Exception:
        pass
    return False


def add_urn_property(eaf, urn_value: str):
    # Use the correct API for header properties
    eaf.add_property("URN", urn_value)


def write_rows_csv(rows: List[Dict], out_csv: str):
    fieldnames = ["audio_file","segment_index","start_ms","end_ms","start_s","end_s","segment_path","text"]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {out_csv} with {len(rows)} rows.")
# ------------------------------
# CLI
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Transcribe audio to CSV and optionally produce ELAN .eaf (single file or batch).")

    # Input modes
    group_in = ap.add_mutually_exclusive_group(required=True)
    group_in.add_argument("--wav", help="Path to input WAV file (single-file mode).")
    group_in.add_argument("--from-csv", help="Use an existing segments CSV (skip transcription; single-file mode).")
    group_in.add_argument("--batch-dir", help="Process all WAVs in this directory (batch mode).")

    # Common ASR args
    ap.add_argument("--segment-ms", type=int, default=20000, help="Segment length in ms (default: 20000).")
    ap.add_argument(
        "--model", default="sjhuskey/enenlhet-wav2vec2-model", help="Model ID or path."
    )

    # Single-file outputs
    ap.add_argument("--out-csv", help="Path to write CSV (single-file).")
    ap.add_argument("--segment-dir", default="segments", help="Directory for temp segments (single-file).")

    # Batch outputs (templates)
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories in --batch-dir.")
    ap.add_argument("--out-csv-template", default="out/csv/{stem}.csv", help="CSV path template for batch mode.")
    ap.add_argument("--segment-dir-template", default="work/segments/{stem}", help="Segment dir template for batch mode.")
    ap.add_argument("--align-to-eaf-template",
    help="Batch: template to locate an existing EAF per WAV (e.g., data/eaf/{stem}.eaf) for align-to-EAF mode.")


    # EAF options
    # Align-to-EAF mode (use existing EAF time spans instead of fixed chunks)
    ap.add_argument("--align-to-eaf", help="Use time spans from this EAF to drive segmentation + transcription.")
    ap.add_argument("--align-tiers", help="Comma-separated list of EAF tier names to use (default: all time-alignable).")
    ap.add_argument("--min-ms", type=int, default=1, help="Skip windows shorter than this many ms (default: 1).")
    ap.add_argument("--to-eaf", action="store_true", help="Also write an ELAN .eaf.")
    eaf_mode = ap.add_mutually_exclusive_group()
    eaf_mode.add_argument("--new-eaf", help="Create a NEW .eaf at this path (single-file).")
    eaf_mode.add_argument("--merge-eaf", help="Merge an ASR tier into EXISTING .eaf at this path (single-file; requires --out-eaf).")
    ap.add_argument("--out-eaf", help="Output .eaf path when merging or to override default (single-file).")
    ap.add_argument("--tier-name", default="ASR", help="Tier name (default: ASR).")
    ap.add_argument("--participant", default=None, help="Optional ELAN participant attribute for the tier.")
    ap.add_argument("--ling-type", default="default-lt", help="Linguistic type name (created if missing).")
    ap.add_argument("--absolute-media", action="store_true", help="Use absolute path for media in new EAF (default: relative).")
    ap.add_argument("--urn", help='Explicit URN value to store as PROPERTY NAME="URN".')
    ap.add_argument("--no-generate-urn", action="store_true",
                    help="Do not auto-generate a URN when creating a NEW EAF.")
    ap.add_argument("--ensure-urn-on-merge", action="store_true",
                    help="If merging and the target EAF lacks a URN, add one.")
    ap.add_argument(
        "--urn-namespace",
        default="urn:enenlhet-asr-elan-eaf",
        help="Namespace used when auto-generating URNs (default: urn:enenlhet-asr-elan-eaf).",
    )
    ap.add_argument("--pause-based", action="store_true",
                help="Segment by pauses instead of fixed-length chunks.")
    ap.add_argument("--min-pause-ms", type=int, default=150,
                    help="Pause length (ms) that creates a boundary (default: 150).")
    ap.add_argument("--silence-thresh", type=int, default=None,
                    help="Absolute silence dBFS threshold (e.g., -40). If not set, use audio.dBFS - rel_thresh_db.")
    ap.add_argument("--rel-thresh-db", type=int, default=16,
                    help="Relative threshold below average loudness (default: 16 dB). Used if --silence-thresh is not set.")
    ap.add_argument("--keep-silence-ms", type=int, default=0,
                    help="How much silence (ms) to keep at each side of a chunk (default: 0).")
    ap.add_argument("--max-chunk-ms", type=int, default=0,
                    help="Optional: if a continuous region is very long, split it into sub-chunks of this length (0=off).")


    # Batch EAF templates
    ap.add_argument("--new-eaf-template", default="out/eaf/{stem}.eaf", help="NEW EAF path template (batch).")
    ap.add_argument("--`-template", help="EXISTING EAF path template to merge into (batch).")
    ap.add_argument("--out-eaf-template", help="Merged EAF output path template (batch).")

    args = ap.parse_args()

    # Load model once for batch speed (if we will transcribe anything)
    print("Preparing model...")
    processor = model = device = None
    will_transcribe = bool(args.wav or args.batch_dir) and not args.from_csv
    if will_transcribe:
        processor, model, device = load_model(args.model)

    align_mode = bool(args.align_to_eaf)

    # ---------------- Single-file mode ----------------
    if args.wav or args.from_csv or align_mode:

        # Build/obtain rows
        print("Processing single file...")
        if args.from_csv:
            rows = read_segments_csv(args.from_csv)
            wav_for_media = rows[0]["audio_file"] if rows else None

        elif align_mode:
            # Resolve WAV source: --wav takes precedence; else try to read from the EAF
            wav_for_media = args.wav or resolve_media_from_eaf(args.align_to_eaf)
            if not wav_for_media or not Path(wav_for_media).exists():
                raise SystemExit(
                    "Align-to-EAF mode: could not resolve a usable WAV. Provide --wav or ensure EAF media is resolvable."
                )
            if processor is None:
                processor, model, device = load_model(args.model)

            rows = transcribe_eaf_spans_to_rows(
                wav_path=wav_for_media,
                eaf_path=args.align_to_eaf,
                processor=processor,
                model=model,
                device=device,
                tiers=args.align_tiers,
                min_ms=args.min_ms,
            )

            # If the user passed --out-csv, persist the CSV (handy for audits)
            if args.out_csv:
                fieldnames = [
                    "audio_file",
                    "segment_index",
                    "start_ms",
                    "end_ms",
                    "start_s",
                    "end_s",
                    "segment_path",
                    "text",
                ]
                Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
                with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        elif args.pause_based:
            if processor is None:
                processor, model, device = load_model(args.model)

            windows = pause_based_windows(
                wav_path=args.wav,
                min_pause_ms=args.min_pause_ms,
                silence_thresh=args.silence_thresh,
                rel_thresh_db=args.rel_thresh_db,
                keep_silence_ms=args.keep_silence_ms,
                max_chunk_ms=args.max_chunk_ms,
            )
            if not windows:
                raise SystemExit("No speech windows detected. Try lowering --silence-thresh (e.g., -50) or --rel-thresh-db, or check the audio.")
            rows = transcribe_windows_to_rows(
                wav_path=args.wav,
                windows=windows,
                processor=processor,
                model=model,
                device=device,
            )
            wav_for_media = args.wav

        else:
            # original fixed-length segmentation path
            if not args.out_csv:
                raise SystemExit(
                    "--out-csv is required in single-file mode when using --wav (non-align mode)."
                )
            rows = transcribe_to_csv(
                wav_path=args.wav,
                out_csv=args.out_csv,
                processor=processor,
                model=model,
                device=device,
                segment_dir=args.segment_dir,
                segment_ms=args.segment_ms,
            )
            wav_for_media = args.wav

        # Optional EAF writing (new or merge) — unchanged:
        if not args.to_eaf:
            return
        print("Writing EAF...")
        if args.new_eaf:
            out_eaf = args.out_eaf or args.new_eaf
            build_new_eaf(
                segments=rows,
                wav_path=wav_for_media,
                out_eaf_path=out_eaf,
                tier_name=args.tier_name,
                ling_type=args.ling_type,
                participant=args.participant,
                relative_media=not args.absolute_media,
                urn=args.urn,
                generate_urn=(not args.no_generate_urn),
                urn_namespace=args.urn_namespace,
            )
        elif args.merge_eaf:
            out_target = args.out_eaf
            if not out_target:
                p = Path(args.merge_eaf)
                out_target = str(p.with_name(p.stem + "_with_asr").with_suffix(".eaf"))
            merge_into_existing_eaf(
                existing_eaf_path=args.merge_eaf,
                segments=rows,
                out_eaf_path=out_target,
                tier_name=args.tier_name,
                ling_type=args.ling_type,
                participant=args.participant,
                ensure_urn_on_merge=args.ensure_urn_on_merge,
                urn=args.urn,
                urn_namespace=args.urn_namespace,
            )
        else:
            raise SystemExit("If --to-eaf is set, provide either --new-eaf or --merge-eaf.")
        return

    # ---------------- Batch mode ----------------
    wav_paths = find_wavs(args.batch_dir, args.recursive)
    if not wav_paths:
        print("No WAV files found.")
        return
    print(f"Found {len(wav_paths)} WAV files in batch directory.")
    print("Processing batch...")
    for wav_path in wav_paths:
        stem = Path(wav_path).stem
        out_csv = expand_template(args.out_csv_template, wav_path)
        seg_dir = expand_template(args.segment_dir_template, wav_path)

        # Ensure model is loaded once for batch when needed
        if processor is None:
            processor, model, device = load_model(args.model)

        # ---- choose segmentation mode for batch ----
        if args.align_to_eaf_template:
            # Align-to-EAF per file
            eaf_in = expand_template(args.align_to_eaf_template, wav_path)
            if not Path(eaf_in).exists():
                print(f"[WARN] Skipping {wav_path} (no EAF at {eaf_in})")
                continue

            # Prefer WAV from CLI batch, but you could resolve from EAF if desired:
            wav_for_media = wav_path

            rows = transcribe_eaf_spans_to_rows(
                wav_path=wav_for_media,
                eaf_path=eaf_in,
                processor=processor,
                model=model,
                device=device,
                tiers=args.align_tiers,
                min_ms=args.min_ms,
            )
            write_rows_csv(rows, out_csv)

        elif args.pause_based:
            # Pause-based windows per file
            windows = pause_based_windows(
                wav_path=wav_path,
                min_pause_ms=args.min_pause_ms,
                silence_thresh=args.silence_thresh,
                rel_thresh_db=args.rel_thresh_db,
                keep_silence_ms=args.keep_silence_ms,
                max_chunk_ms=args.max_chunk_ms,
            )
            if not windows:
                print(f"[WARN] No speech windows for {wav_path}; skipping.")
                continue

            rows = transcribe_windows_to_rows(
                wav_path=wav_path,
                windows=windows,
                processor=processor,
                model=model,
                device=device,
            )
            write_rows_csv(rows, out_csv)
            wav_for_media = wav_path

        else:
            # Fixed-length (legacy)
            rows = transcribe_to_csv(
                wav_path=wav_path,
                out_csv=out_csv,
                processor=processor,
                model=model,
                device=device,
                segment_dir=seg_dir,
                segment_ms=args.segment_ms,
            )
            wav_for_media = wav_path

        # ---- EAF emission ----
        if not args.to_eaf:
            continue
        print(f"Writing EAF for {stem}...")
        if args.merge_eaf_template:
            if not args.out_eaf_template:
                raise SystemExit("--merge-eaf-template requires --out-eaf-template in batch mode.")
            existing_eaf = expand_template(args.merge_eaf_template, wav_path)
            out_eaf = expand_template(args.out_eaf_template, wav_path)
            merge_into_existing_eaf(
                existing_eaf_path=existing_eaf,
                segments=rows,
                out_eaf_path=out_eaf,
                tier_name=args.tier_name,
                ling_type=args.ling_type,
                participant=args.participant,
                ensure_urn_on_merge=args.ensure_urn_on_merge,
                urn=args.urn,
                urn_namespace=args.urn_namespace,
            )
        else:
            out_eaf = expand_template(args.new_eaf_template, wav_path)
            build_new_eaf(
                segments=rows,
                wav_path=wav_for_media,
                out_eaf_path=out_eaf,
                tier_name=args.tier_name,
                ling_type=args.ling_type,
                participant=args.participant,
                relative_media=not args.absolute_media,
                urn=args.urn,
                generate_urn=(not args.no_generate_urn),
                urn_namespace=args.urn_namespace,
            )


if __name__ == "__main__":
    main()
