import argparse, os, math, random, subprocess
from pathlib import Path

import pandas as pd
import cv2
import numpy as np


def read_all_frames(mp4_path: str):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def read_last_frame(mp4_path: str):
    frames = read_all_frames(mp4_path)
    if len(frames) == 0:
        raise RuntimeError(f"0 frames in {mp4_path}")
    return frames[-1]


def write_mp4(frames, out_path: Path, fps: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) == 0:
        raise RuntimeError(f"merge produced 0 frames for {out_path}")
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter failed for {out_path}")
    for fr in frames:
        vw.write(fr)
    vw.release()


def get_col(r, *names):
    for n in names:
        if n in r and pd.notna(r[n]):
            return r[n]
    raise KeyError(names)


def grab_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        raise RuntimeError(f"failed to read frame {frame_idx} from {video_path}")
    return fr


def build_chunk_frame_indices(raw_start, raw_end, fps, down_fps, stride, frames, chunk_id):
    step = max(1, fps // down_fps)  # raw->downsample
    need = 1 + (frames - 1) * stride  # how many downsampled frames needed for fixed stride
    raw_len = max(raw_end - raw_start, 1)
    T_down = int(math.ceil(raw_len / step))

    # non-overlapping chunks in downsampled timeline, but ensure coverage by adding last chunk
    starts = list(range(0, max(T_down, 1), need))
    last_start = max(0, T_down - need)
    if last_start not in starts:
        starts.append(last_start)
    starts = sorted(starts)

    if chunk_id >= len(starts):
        return None, None, None, None

    s_down = starts[chunk_id]
    idx_down = (s_down + np.arange(frames, dtype=np.int64) * stride)

    # clamp into [0, T_down-1]
    idx_down = np.clip(idx_down, 0, max(T_down - 1, 0))

    # convert to raw frame indices
    idx_raw = raw_start + idx_down * step
    idx_raw = np.clip(idx_raw, raw_start, max(raw_end - 1, raw_start))
    return idx_raw.tolist(), len(starts), step, need


def run_one_chunk(args, base_row, outdir: Path, orig_row: int, chunk_id: int, init_frame_img, seed: int):
    # resolve paths + bounds
    fps = int(base_row["fps"])
    raw_start = int(base_row["start_frame"])
    raw_end = int(base_row["end_frame"])

    normal_path = get_col(base_row, "normal_path", "normal_video", "normal")
    trans_path  = get_col(base_row, "transparent_path", "transparent_video", "transparent")

    idx_raw, num_chunks, step, need = build_chunk_frame_indices(
        raw_start, raw_end, fps, args.down_fps, args.stride, args.frames, chunk_id
    )
    if idx_raw is None:
        return None, None, None

    # transparent frames (already fixed-stride selected)
    trans_frames = []
    cap_t = cv2.VideoCapture(trans_path)
    if not cap_t.isOpened():
        raise RuntimeError(f"cannot open {trans_path}")
    for fr_idx in idx_raw:
        cap_t.set(cv2.CAP_PROP_POS_FRAMES, int(fr_idx))
        ok, fr = cap_t.read()
        if not ok or fr is None:
            # pad with last good frame if read fails
            if len(trans_frames) == 0:
                raise RuntimeError(f"failed reading first frame {fr_idx} from {trans_path}")
            fr = trans_frames[-1].copy()
        trans_frames.append(fr)
    cap_t.release()

    # normal tmp: repeat init frame to match length (prevents "1 vs N frames" errors)
    norm_frames = [init_frame_img.copy() for _ in range(args.frames)]

    # write tmp videos + meta (one-row CSV)
    trans_tmp = outdir / f"orig{orig_row}_chunk{chunk_id}_trans_tmp.mp4"
    norm_tmp  = outdir / f"orig{orig_row}_chunk{chunk_id}_normal_tmp.mp4"
    write_mp4(trans_frames, trans_tmp, args.down_fps)
    write_mp4(norm_frames, norm_tmp, args.down_fps)

    meta_tmp = outdir / f"orig{orig_row}_chunk{chunk_id}_meta_tmp.csv"
    row = dict(base_row)
    row["normal_path"] = str(norm_tmp)
    row["transparent_path"] = str(trans_tmp)
    row["start_frame"] = 0
    row["end_frame"] = args.frames
    row["fps"] = args.down_fps
    pd.DataFrame([row]).to_csv(meta_tmp, index=False)

    out_base = outdir / f"orig{orig_row}_chunk{chunk_id}_out.mp4"

    cmd = [
        "python", args.inpaint_py,
        "--prompt", args.prompt,
        "--model_path", args.model_path,
        "--generate_type", "i2v_inpainting",
        "--inpainting_branch", args.branch_path,
        "--inpainting_mask_meta", str(meta_tmp),
        "--inpainting_sample_id", "0",
        "--inpainting_frames", str(args.frames),
        "--down_sample_fps", str(args.down_fps),
        "--num_inference_steps", str(args.steps),
        "--guidance_scale", str(args.guidance),
        "--seed", str(seed),
        "--output_path", str(out_base),
    ]
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # resolve actual output name (some scripts append _fps{down_fps})
    cands = [
        out_base,
        outdir / f"orig{orig_row}_chunk{chunk_id}_out_fps{args.down_fps}.mp4",
        outdir / f"orig{orig_row}_chunk{chunk_id}_out_fps{args.down_fps}_fps{args.down_fps}.mp4",
        out_base.with_name(out_base.stem + f"_fps{args.down_fps}.mp4"),
        out_base.with_name(out_base.stem + f"_fps{args.down_fps}_fps{args.down_fps}.mp4"),
    ]
    out_mp4 = None
    for c in cands:
        if c.exists() and c.stat().st_size > 1024:
            out_mp4 = c
            break
    if out_mp4 is None:
        raise RuntimeError("output mp4 not found. looked for: " + ", ".join(map(str,cands)))

    return out_mp4, meta_tmp, num_chunks


def merge_chunks(chunk_paths, merged_path: Path, fps: int):
    chunk_paths = chunk_paths[:2]  # ONLY merge chunk0+chunk1

    merged = []
    for i, p in enumerate(chunk_paths):
        frames = read_all_frames(str(p))
        if len(frames) == 0:
            raise RuntimeError(f"0 frames in {p}")
        if i > 0:
            frames = frames[1:]  # drop duplicate init frame
        merged.extend(frames)
    write_mp4(merged, merged_path, fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_in", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--branch_path", required=True)
    ap.add_argument("--inpaint_py", required=True)
    ap.add_argument("--prompt", default="a robot arm closing a drawer in a kitchen")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--down_fps", type=int, default=8)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--num_demos", type=int, default=5)
    ap.add_argument("--seed", type=int, default=-1, help="-1 => random each run; otherwise deterministic")
    ap.add_argument("--keep_chunks", action="store_true")
    args = ap.parse_args()
    args.seed = -1  # force random demos each run

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv_in)
    if "caption" in df.columns:
        df = df[df["caption"].astype(str).str.len() > 50].reset_index(drop=True)

    rng = random.Random(None if args.seed == -1 else args.seed)
    picked = rng.sample(range(len(df)), min(args.num_demos, len(df)))
    (outdir / f"picked_rows_seed{args.seed}.txt").write_text("\n".join(map(str, picked)) + "\n")
    print("Picked orig_row demos:", picked)

    for orig_row in picked:
        base_row = df.iloc[orig_row].to_dict()

        # chunk0 init from normal[0]
        normal_path = get_col(base_row, "normal_path", "normal_video", "normal")
        init0 = grab_frame(normal_path, int(base_row["start_frame"]))
        chunk_paths = []

        seed0 = (args.seed if args.seed != -1 else rng.randrange(0, 10**9))
        cur_seed = seed0

        # run sequential chunks, chaining init from last generated frame
        chunk_id = 0
        init_frame = init0
        num_chunks = None

        while True:
            out_mp4, meta_tmp, num_chunks = run_one_chunk(
                args, base_row, outdir, orig_row, chunk_id, init_frame, cur_seed
            )
            if out_mp4 is None:
                break
            chunk_paths.append(out_mp4)

            # next init = last frame of this output
            init_frame = read_last_frame(str(out_mp4))
            chunk_id += 1
            cur_seed += 1

            if chunk_id >= 2:
                break

        if len(chunk_paths) == 0:
            print(f"SKIP orig_row {orig_row}: no chunks")
            continue

        merged_path = outdir / f"orig{orig_row}_merged_fps{args.down_fps}.mp4"
        merge_chunks(chunk_paths, merged_path, args.down_fps)
        print("MERGED:", merged_path)

        if not args.keep_chunks:
            for p in list(outdir.glob(f"orig{orig_row}_chunk*_out*.mp4")):
                p.unlink(missing_ok=True)
            for p in list(outdir.glob(f"orig{orig_row}_chunk*_trans_tmp.mp4")):
                p.unlink(missing_ok=True)
            for p in list(outdir.glob(f"orig{orig_row}_chunk*_normal_tmp.mp4")):
                p.unlink(missing_ok=True)
            for p in list(outdir.glob(f"orig{orig_row}_chunk*_meta_tmp.csv")):
                p.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
