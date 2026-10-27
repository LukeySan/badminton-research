"""
Mark your court on a frame of the video, so shuttles on other courts can be ignored.

Usage:
    python pick_court.py ../clipped_vid.mp4
    python pick_court.py ../clipped_vid.mp4 --time 10    # use the frame at 10 seconds

Click the 4 corners of YOUR court's outer lines, in this order:
    1. far-left   2. far-right   3. near-right   4. near-left
If a corner is outside the video (camera too close), click in the gray margin
where the corner would be: follow the sideline and baseline past the white
frame edge and click where they'd meet.
Keys: u = undo last click, r = reset, Enter/s = save, q/Esc = quit without saving.
Saves court.json (corner positions in the video's own pixel coordinates).
"""

import argparse
import json

import cv2
import numpy as np

LABELS = ["far-left", "far-right", "near-right", "near-left"]
MAX_DISPLAY_WIDTH = 1400


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--time", type=float, default=1.0, help="which second of the video to show")
    parser.add_argument("--out", default="court.json")
    parser.add_argument("--pad", type=float, default=0.3,
                        help="size of the gray margin around the video, for corners outside the frame")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Couldn't read a frame from the video; check the path.")

    h, w = frame.shape[:2]
    # Gray margin around the bottom and sides, so corners that fall outside the
    # video (camera too close) can still be clicked where they WOULD be.
    pad_x, pad_b = int(w * args.pad), int(h * args.pad)
    padded = cv2.copyMakeBorder(frame, 0, pad_b, pad_x, pad_x, cv2.BORDER_CONSTANT, value=(60, 60, 60))
    cv2.rectangle(padded, (pad_x, 0), (pad_x + w - 1, h - 1), (255, 255, 255), 2)
    pw = padded.shape[1]
    scale = min(1.0, MAX_DISPLAY_WIDTH / pw)  # shrink big (e.g. 4K) frames to fit the screen
    base = cv2.resize(padded, (int(pw * scale), int(padded.shape[0] * scale))) if scale < 1 else padded
    points = []  # stored in original video pixels (can be outside the frame)

    def to_disp(x, y):
        return int((x + pad_x) * scale), int(y * scale)

    def redraw():
        img = base.copy()
        disp = [to_disp(x, y) for x, y in points]
        for i, p in enumerate(disp):
            cv2.circle(img, p, 6, (0, 255, 255), -1)
            cv2.putText(img, LABELS[i], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(disp) > 1:
            cv2.polylines(img, [np.array(disp)], len(disp) == 4, (0, 255, 0), 2)
        msg = (f"Click: {LABELS[len(points)]}" if len(points) < 4
               else "Done - press Enter to save, u to undo")
        cv2.putText(img, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(img, msg, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Mark your court", img)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((round(x / scale) - pad_x, round(y / scale)))
            redraw()

    cv2.namedWindow("Mark your court")
    cv2.setMouseCallback("Mark your court", on_click)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("u"),) and points:
            points.pop(); redraw()
        elif key == ord("r"):
            points.clear(); redraw()
        elif key in (13, 10, ord("s")) and len(points) == 4:
            with open(args.out, "w") as fh:
                json.dump({"video_size": [w, h], "corners": dict(zip(LABELS, points))}, fh, indent=2)
            print(f"Saved court corners to {args.out}")
            break
        elif key in (ord("q"), 27):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()