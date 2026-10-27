"""
First model test: run YOLO pose on a badminton clip and save a video
with player skeletons drawn on it.

Usage:
    python pose_test.py my_clip.mp4
    python pose_test.py my_clip.mp4 --device cpu      # if the GPU option errors
    python pose_test.py my_clip.mp4 --stride 2        # process every 2nd frame (faster)
"""

import argparse
import time

import torch
from ultralytics import YOLO


def pick_device():
    if torch.cuda.is_available():
        return "cuda"   # NVIDIA GPU (your gaming PC)
    if torch.backends.mps.is_available():
        return "mps"    # Apple Silicon Mac GPU
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="path to your badminton clip")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu (auto by default)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--model", default="yolo11n-pose.pt",
                        help="yolo11n-pose.pt (fast), yolo11s-pose.pt, yolo11m-pose.pt (more accurate)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="resolution the model sees; 1280 helps with small, far-away players")
    parser.add_argument("--conf", type=float, default=0.4, help="minimum detection confidence")
    args = parser.parse_args()

    device = args.device or pick_device()
    print(f"Running on: {device}")

    # Pretrained pose model; downloads automatically the first time.
    model = YOLO(args.model)
    print(f"Model: {args.model}, image size: {args.imgsz}, confidence: {args.conf}")

    start = time.time()
    frames = 0
    people_counts = []

    # stream=True processes frame by frame instead of loading the whole video into memory.
    for result in model.predict(
        source=args.video,
        device=device,
        save=True,              # writes the annotated video
        stream=True,
        vid_stride=args.stride,
        imgsz=args.imgsz,
        conf=args.conf,         # ignore low-confidence detections
        verbose=False,
    ):
        frames += 1
        people_counts.append(len(result.boxes))
        if frames % 100 == 0:
            print(f"  {frames} frames processed...")

    elapsed = time.time() - start
    print(f"\nDone: {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} frames/sec)")
    if people_counts:
        print(f"People detected per frame: avg {sum(people_counts) / len(people_counts):.1f}, "
              f"max {max(people_counts)}")
    print(f"Annotated video saved in: {result.save_dir}")


if __name__ == "__main__":
    main()