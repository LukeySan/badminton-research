"""
Train and use a machine-learning rally detector (your first trained model).

Instead of hand-written rules, a classifier learns from your labels which
frames are "in a rally", using features computed from TrackNet's shuttle track
(speed, direction, height, visibility, and how they change over time).

1) Describe your labeled matches in matches.json:
    [
      {"name": "match1",
       "csv": "prediction/clipped_vid_ball.csv",
       "video": "prediction/clipped_vid.mp4",
       "court": "results/match1/court.json",
       "labels": "results/match1/labels.csv"},
      {"name": "match3", ...}
    ]

2) Train + honest test (each match is scored by a model that never saw it):
    python rally_model.py train matches.json

3) Use the final model (trained on all matches) on a new video:
    python rally_model.py predict --csv prediction/new_ball.csv --video prediction/new.mp4 \
        --court results/new/court.json --out results/new/rallies.json

Needs: pip install scikit-learn joblib
"""

import argparse
import csv
import json
import os

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from rally_detect import load_corners, court_zone, in_zone, rule_detect
from score_rallies import match as match_rallies

MODEL_FILE = "rally_model.joblib"


# ---------------------------------------------------------------- features

def video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps <= 0:
        raise SystemExit(f"Couldn't read the frame rate of {path}")
    return fps


def make_features(csv_path, video_path, court_path, margin=40, use_rules=True):
    """One row of features per video frame, normalized by court size so they
    mean the same thing in any video resolution or camera distance."""
    fps = video_fps(video_path)
    df = pd.read_csv(csv_path).sort_values("Frame").reset_index(drop=True)
    xs, ys = df["X"].to_numpy(float), df["Y"].to_numpy(float)
    vis = (df["Visibility"] == 1).to_numpy()

    corners = load_corners(court_path)
    fl, fr, nr, nl = [np.array(c, float) for c in corners]
    zone = court_zone(corners)
    vis = np.array([v and in_zone(zone, x, y, margin) for v, x, y in zip(vis, xs, ys)])

    width = np.linalg.norm(nr - nl)                         # near baseline, px
    near_y, far_y = (nl[1] + nr[1]) / 2, (fl[1] + fr[1]) / 2
    height = max(near_y - far_y, 1.0)                       # court depth on screen, px
    center_x = (nl[0] + nr[0]) / 2

    # Position relative to the court: xn 0 = centre line, yn 0 = near baseline,
    # 1 = far baseline, >1 = in the air above the far half.
    xn = np.where(vis, (xs - center_x) / width, np.nan)
    yn = np.where(vis, (near_y - ys) / height, np.nan)

    n = len(df)
    vx = np.full(n, np.nan); vy = np.full(n, np.nan)
    last = None
    for i in range(n):
        if not vis[i]:
            continue
        if last is not None and i - last <= 3:
            vx[i] = (xs[i] - xs[last]) / (i - last) / width
            vy[i] = (ys[i] - ys[last]) / (i - last) / width
        last = i
    speed = np.hypot(vx, vy)

    F = pd.DataFrame({"vis": vis.astype(float), "xn": xn, "yn": yn,
                      "vx": vx, "vy": vy, "speed": speed})
    still = (vis & (np.nan_to_num(speed, nan=1.0) < 0.002)).astype(float)
    turn = np.zeros(n)                                     # sharp direction changes
    for i in range(3, n - 3):
        if vis[i - 3] and vis[i] and vis[i + 3]:
            a = np.array([xs[i] - xs[i - 3], ys[i] - ys[i - 3]])
            b = np.array([xs[i + 3] - xs[i], ys[i + 3] - ys[i]])
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0.01 * width and nb > 0.01 * width:
                turn[i] = float(np.dot(a, b) / (na * nb) < -0.17)   # > ~100 degrees
    F["still"], F["turn"] = still, turn

    # Context: the same signals summarized over windows around each frame.
    for sec in (0.25, 0.5, 1.0, 2.0):
        w = max(3, int(fps * sec)) | 1
        r = F.rolling(w, center=True, min_periods=1)
        F[f"vis_c{sec}"] = r["vis"].mean()
        F[f"spd_mean_c{sec}"] = r["speed"].mean()
        F[f"spd_max_c{sec}"] = r["speed"].max()
        F[f"spd_med_c{sec}"] = r["speed"].median()
        F[f"x_std_c{sec}"] = r["xn"].std()
        F[f"y_std_c{sec}"] = r["yn"].std()
        F[f"y_min_c{sec}"] = r["yn"].min()
        F[f"y_max_c{sec}"] = r["yn"].max()
        F[f"vy_mean_c{sec}"] = r["vy"].mean()
        F[f"still_c{sec}"] = r["still"].mean()
        F[f"turns_c{sec}"] = r["turn"].sum()

    # Looking only backward / only forward helps find where rallies start and end.
    rev = F[::-1]
    for sec in (1.0, 3.0):
        w = max(2, int(fps * sec))
        past = F.rolling(w, min_periods=1)
        fut = rev.rolling(w, min_periods=1)
        for col in ("speed", "vis", "still", "turn"):
            agg = "sum" if col == "turn" else "mean"
            F[f"{col}_past{sec}"] = getattr(past[col], agg)()
            F[f"{col}_fut{sec}"] = getattr(fut[col], agg)()[::-1].to_numpy()

    if use_rules:
        add_rule_features(F, csv_path, video_path, court_path, margin, fps)

    return F, fps


def add_rule_features(F, csv_path, video_path, court_path, margin, fps):
    """Hybrid: give the model the hand-written rules' opinion, so it can learn
    when to trust them (they rarely miss rallies) and when to overrule them
    (pickups and tosses they mistake for play)."""
    r = rule_detect(csv_path, video_path, court_path, margin=margin)
    n = len(F)
    in_rally = np.zeros(n)
    since_start = np.full(n, 10.0)   # seconds since the rules' rally began (capped)
    until_end = np.full(n, 10.0)     # seconds until the rules' rally ends (capped)
    seg_len = np.zeros(n)
    for s, e in r["segments"]:
        in_rally[s:e + 1] = 1
        idx = np.arange(s, e + 1)
        since_start[s:e + 1] = np.minimum((idx - s) / fps, 10)
        until_end[s:e + 1] = np.minimum((e - idx) / fps, 10)
        seg_len[s:e + 1] = (e - s) / fps
    landed = r["landed"].astype(float)
    since_landed = np.full(n, 10.0)
    last = None
    for i in range(n):
        if landed[i]:
            last = i
        if last is not None:
            since_landed[i] = min((i - last) / fps, 10)

    F["rule_in_rally"] = in_rally
    F["rule_moving"] = r["moving"].astype(float)
    F["rule_landed"] = landed
    F["rule_since_start"] = since_start
    F["rule_until_end"] = until_end
    F["rule_seg_len"] = seg_len
    F["rule_since_landed"] = since_landed
    for sec in (0.5, 2.0):
        w = max(3, int(fps * sec)) | 1
        F[f"rule_moving_c{sec}"] = F["rule_moving"].rolling(w, center=True, min_periods=1).mean()
        F[f"rule_landed_c{sec}"] = F["rule_landed"].rolling(w, center=True, min_periods=1).mean()


def frame_labels(labels_path, n_frames, fps):
    y = np.zeros(n_frames, dtype=int)
    with open(labels_path) as fh:
        for row in csv.DictReader(fh):
            if row.get("end"):
                a = int(float(row["start"]) * fps)
                b = int(float(row["end"]) * fps)
                y[max(a, 0):min(b + 1, n_frames)] = 1
    return y


# ---------------------------------------------------------------- rallies from probabilities

def to_rallies(prob, fps, threshold=0.5, smooth=0.4, merge_gap=0.7, min_len=1.5):
    w = max(3, int(fps * smooth)) | 1
    p = pd.Series(prob).rolling(w, center=True, min_periods=1).median().to_numpy()
    on = np.flatnonzero(p > threshold)
    segs, start, prev = [], None, None
    for f in on:
        if start is None:
            start = prev = f
        elif f - prev > merge_gap * fps:
            segs.append((start, prev)); start = f
        prev = f
    if start is not None:
        segs.append((start, prev))
    return [(round(s / fps, 2), round(e / fps, 2)) for s, e in segs if (e - s) / fps >= min_len]


def summarize(detected, truth, tol=1.0):
    m = match_rallies(detected, truth)
    se = [abs(detected[i][0] - truth[j][0]) for i, j in m]
    ee = [abs(detected[i][1] - truth[j][1]) for i, j in m]
    return {
        "real": len(truth), "found": len(m),
        "exact": sum(1 for a, b in zip(se, ee) if a <= tol and b <= tol),
        "missed": len(truth) - len(m), "fake": len(detected) - len(m),
        "start_err": np.mean(se) if se else float("nan"),
        "end_err": np.mean(ee) if ee else float("nan"),
        "late": sum(1 for i, j in m if detected[i][1] - truth[j][1] > tol),
    }


def print_summary(name, s):
    print(f"  {name:<16} found {s['found']}/{s['real']}  accurate {s['exact']}/{s['real']}  "
          f"missed {s['missed']}  fake {s['fake']}  late {s['late']}  "
          f"start err {s['start_err']:.1f}s  end err {s['end_err']:.1f}s")


def save_rallies(path, rallies, fps):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = [{"id": i + 1, "start": s, "end": e, "duration": round(e - s, 2)}
           for i, (s, e) in enumerate(rallies)]
    with open(path, "w") as fh:
        json.dump({"fps": fps, "rallies": out}, fh, indent=2)


def new_model():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=31, l2_regularization=1.0,
                                          random_state=0)


# ---------------------------------------------------------------- commands

def cmd_train(args):
    with open(args.matches) as fh:
        matches = json.load(fh)
    data = {}
    for m in matches:
        print(f"Computing features for {m['name']}...")
        F, fps = make_features(m["csv"], m["video"], m["court"], use_rules=not args.no_rules)
        y = frame_labels(m["labels"], len(F), fps)
        with open(m["labels"]) as fh:
            truth = sorted((float(r["start"]), float(r["end"])) for r in csv.DictReader(fh) if r.get("end"))
        data[m["name"]] = (F, y, fps, truth, m)
        print(f"  {len(F)} frames, {100 * y.mean():.0f}% labeled as in-rally, {len(truth)} rallies")

    names = list(data)
    if len(names) >= 2:
        print("\nHonest test: each match scored by a model trained only on the OTHER matches")
        totals, held_out = [], []
        for test in names:
            X = pd.concat([data[n][0] for n in names if n != test])
            Y = np.concatenate([data[n][1] for n in names if n != test])
            model = new_model().fit(X, Y)
            F, y, fps, truth, m = data[test]
            prob = model.predict_proba(F)[:, 1]
            frame_acc = ((prob > 0.5) == y).mean()
            rallies = to_rallies(prob, fps, threshold=args.threshold, smooth=args.smooth,
                                 merge_gap=args.merge_gap, min_len=args.min_len)
            out = os.path.join(os.path.dirname(m["labels"]) or ".", f"{test}_rallies_model.json")
            save_rallies(out, rallies, fps)
            r = rule_detect(m["csv"], m["video"], m["court"])
            rule_rallies = [(round(a / r["fps"], 2), round(b / r["fps"], 2)) for a, b in r["segments"]]
            print_summary(f"{test} model", summarize(rallies, truth))
            print_summary(f"{test} rules", summarize(rule_rallies, truth))
            print(f"  {'':<16} frame accuracy {100 * frame_acc:.0f}%   saved {out}\n")
            totals.append((summarize(rallies, truth), summarize(rule_rallies, truth)))
            held_out.append((prob, fps, truth))
        keys = ("real", "found", "exact", "missed", "fake", "late")
        for label, k in (("ALL model", 0), ("ALL rules", 1)):
            t = {key: sum(x[k][key] for x in totals) for key in keys}
            print(f"  {label:<16} accurate {t['exact']}/{t['real']}  missed {t['missed']}  "
                  f"fake {t['fake']}  late {t['late']}")

        # The same held-out predictions, cut at different probability thresholds.
        # Lower = more rallies kept (fewer misses, more fakes); higher = the reverse.
        print("\nThreshold sweep (all matches combined, same honest predictions):")
        print("  threshold  accurate  missed  fake  late")
        for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
            t = {key: 0 for key in keys}
            for prob, fps, truth in held_out:
                det = to_rallies(prob, fps, threshold=thr, smooth=args.smooth,
                                 merge_gap=args.merge_gap, min_len=args.min_len)
                sm = summarize(det, truth)
                for key in keys:
                    t[key] += sm[key]
            mark = "  <- current" if abs(thr - args.threshold) < 1e-9 else ""
            print(f"  {thr:>9}  {t['exact']:>4}/{t['real']}  {t['missed']:>6}  {t['fake']:>4}  {t['late']:>4}{mark}")
        print("  (pick one with --threshold and rerun to save a model with it)")
    else:
        print("\nOnly one match: skipping the honest test (needs 2+ labeled matches).")

    print("\nTraining the final model on all matches...")
    X = pd.concat([data[n][0] for n in names])
    Y = np.concatenate([data[n][1] for n in names])
    model = new_model().fit(X, Y)
    joblib.dump({"model": model, "columns": list(X.columns), "threshold": args.threshold,
                 "use_rules": not args.no_rules,
                 "smooth": args.smooth, "merge_gap": args.merge_gap, "min_len": args.min_len}, MODEL_FILE)
    print(f"Saved {MODEL_FILE}")
    print("(Scores on matches the model trained on are not a fair test; use the ones above.)")


def cmd_predict(args):
    bundle = joblib.load(args.model)
    F, fps = make_features(args.csv, args.video, args.court, use_rules=bundle.get("use_rules", False))
    prob = bundle["model"].predict_proba(F[bundle["columns"]])[:, 1]
    rallies = to_rallies(prob, fps, threshold=args.threshold or bundle["threshold"],
                         smooth=bundle.get("smooth", 0.4), merge_gap=bundle.get("merge_gap", 0.7),
                         min_len=bundle.get("min_len", 1.5))
    save_rallies(args.out, rallies, fps)
    for i, (s, e) in enumerate(rallies, 1):
        print(f"Rally {i:>2}: {s:7.1f}s - {e:7.1f}s  ({e - s:.1f}s)")
    print(f"\nSaved {len(rallies)} rallies to {args.out}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("matches", help="matches.json listing your labeled matches")
    t.add_argument("--threshold", type=float, default=0.5,
                   help="probability above which a frame counts as in-rally")
    t.add_argument("--no-rules", action="store_true",
                   help="train without the rules' opinion as features (the non-hybrid model)")
    t.add_argument("--merge-gap", type=float, default=0.7,
                   help="join pieces closer than this many seconds into one rally")
    t.add_argument("--smooth", type=float, default=0.4,
                   help="seconds of smoothing applied to the frame probabilities")
    t.add_argument("--min-len", type=float, default=1.5,
                   help="drop rallies shorter than this many seconds")
    q = sub.add_parser("predict")
    q.add_argument("--csv", required=True)
    q.add_argument("--video", required=True)
    q.add_argument("--court", required=True)
    q.add_argument("--out", default="rallies.json")
    q.add_argument("--model", default=MODEL_FILE)
    q.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()
    cmd_train(args) if args.cmd == "train" else cmd_predict(args)


if __name__ == "__main__":
    main()