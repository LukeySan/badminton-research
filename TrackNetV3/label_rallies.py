"""
Label the real rally start/end times in a video by pressing keys while it plays.

Usage:
    python label_rallies.py match2.mp4 --out results/match2/labels.csv
    python label_rallies.py match2.mp4 --out results/match2/labels.csv --speed 2   # play 2x faster

Keys (click the video window first so it receives key presses):
    s           mark rally START (the moment the server hits the shuttle)
    e           mark rally END   (shuttle hits the floor, goes out, hits the net for good, or a fault is called)
    space       pause / play
    left/right  jump back / forward 2 seconds   (a / d also work)
    , / .       step back / forward one frame (while paused)
    + / -       faster / slower playback
    u           undo the last mark
    w           save now
    q / Esc     save and quit

The file is saved after every change, so you can stop and resume later:
existing labels in --out are loaded when you start again.
"""

import argparse
import csv
import os

import cv2

MAX_DISPLAY_WIDTH = 1280


def fmt(seconds):
    m, s = divmod(max(seconds, 0), 60)
    return f"{int(m)}:{s:04.1f}"


def load(path):
    rallies = []
    if os.path.exists(path):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                end = row.get("end", "")
                rallies.append([float(row["start"]), float(end) if end else None])
    return rallies


def save(path, rallies):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start", "end"])
        for s, e in sorted(rallies, key=lambda r: r[0]):
            w.writerow([f"{s:.2f}", f"{e:.2f}" if e is not None else ""])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--out", default="labels.csv")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise SystemExit("Couldn't read the video; check the path.")

    rallies = load(args.out)
    if rallies:
        print(f"Loaded {len(rallies)} existing labels from {args.out}")
    history = []
    speed, paused = args.speed, False
    frame_idx, frame = 0, None

    def seek(i):
        nonlocal frame_idx, frame
        frame_idx = max(0, min(total - 1, i))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, f = cap.read()
        if ok:
            frame = f

    seek(0)
    cv2.namedWindow("Label rallies")
    while True:
        if not paused:
            # Skip frames to play faster than 1x without re-seeking every frame.
            steps = max(1, int(round(speed)))
            for _ in range(steps):
                ok, f = cap.read()
                if not ok:
                    paused = True
                    break
                frame, frame_idx = f, frame_idx + 1

        t = frame_idx / fps
        h, w = frame.shape[:2]
        scale = min(1.0, MAX_DISPLAY_WIDTH / w)
        img = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame.copy()

        open_rally = rallies and rallies[-1][1] is None
        in_rally = any(s <= t <= (e if e is not None else t) for s, e in rallies)
        status = "IN RALLY" if in_rally else "break"
        lines = [
            f"{fmt(t)} / {fmt(total / fps)}   speed {speed:g}x   {'PAUSED' if paused else ''}",
            f"rallies labeled: {sum(1 for r in rallies if r[1] is not None)}   now: {status}"
            + ("   (waiting for 'e')" if open_rally else ""),
            "s=start  e=end  space=pause  arrows=+/-2s  ,/.=frame  u=undo  q=quit",
        ]
        for i, line in enumerate(lines):
            y = 28 + i * 26
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
            cv2.putText(img, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 255) if in_rally else (255, 255, 255), 2)
        # Timeline bar with labeled rallies
        bar_y = img.shape[0] - 14
        cv2.rectangle(img, (0, bar_y - 6), (img.shape[1], bar_y + 6), (50, 50, 50), -1)
        for s, e in rallies:
            x1 = int(s / (total / fps) * img.shape[1])
            x2 = int((e if e is not None else t) / (total / fps) * img.shape[1])
            cv2.rectangle(img, (x1, bar_y - 6), (max(x2, x1 + 2), bar_y + 6), (0, 200, 255), -1)
        px = int(t / (total / fps) * img.shape[1])
        cv2.line(img, (px, bar_y - 10), (px, bar_y + 10), (255, 255, 255), 2)
        cv2.imshow("Label rallies", img)

        # Fast speeds skip frames (above); slow speeds wait longer between frames.
        delay = 30 if paused else max(1, int(1000 / (fps * min(speed, 1.0))))
        key = cv2.waitKeyEx(delay)
        if key == -1:
            continue
        k = key & 0xFF

        if k in (ord("q"), 27):
            break
        elif k == ord(" "):
            paused = not paused
        elif k == ord("s"):
            if open_rally:
                print(f"Rally already open since {fmt(rallies[-1][0])}; press 'e' to end it first (or 'u').")
            else:
                rallies.append([t, None]); history.append("s")
                print(f"START {fmt(t)}"); save(args.out, rallies)
        elif k == ord("e"):
            if open_rally and t > rallies[-1][0]:
                rallies[-1][1] = t; history.append("e")
                print(f"END   {fmt(t)}  ({t - rallies[-1][0]:.1f}s)"); save(args.out, rallies)
            else:
                print("No open rally to end (press 's' first), or end is before start.")
        elif k == ord("u") and rallies:
            if rallies[-1][1] is not None:
                rallies[-1][1] = None
                print("Undid last END")
            else:
                rallies.pop()
                print("Undid last START")
            save(args.out, rallies)
        elif k == ord("w"):
            save(args.out, rallies); print(f"Saved to {args.out}")
        elif k in (ord("+"), ord("=")):
            speed = min(8, speed * 2)
        elif k in (ord("-"), ord("_")):
            speed = max(0.25, speed / 2)
        elif k == ord(",") :
            paused = True; seek(frame_idx - 1)
        elif k == ord("."):
            paused = True; seek(frame_idx + 1)
        elif k == ord("a") or key in (2424832, 63234, 65361):   # left arrow (Win / Mac / Linux)
            seek(frame_idx - int(2 * fps))
        elif k == ord("d") or key in (2555904, 63235, 65363):   # right arrow
            seek(frame_idx + int(2 * fps))

    save(args.out, rallies)
    done = [r for r in rallies if r[1] is not None]
    print(f"\nSaved {len(done)} rallies to {args.out}")
    if len(done) < len(rallies):
        print("Note: the last rally has a start but no end.")
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()