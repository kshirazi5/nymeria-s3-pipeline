"""
Run the Nymeria pipeline and the leakage audit on a laptop, no AWS needed.

Steps
1. Download the lightweight metadata and atomic-narration files listed in the
   Nymeria manifest into a local folder (skips files already downloaded).
2. Rebuild the sequence-level feature table and the V2 discomfort proxy label,
   following the notebook.
3. Reproduce the original V2 result, then run a 2x2 ablation and the
   leakage-corrected model comparison (participant-grouped CV).

Usage
    python scripts/run_local.py --manifest Nymeria_download_urls.json
"""
import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedGroupKFold, StratifiedKFold,
                                     cross_validate, train_test_split)

SEED = 42
TEXT_COL = "Describe my atomic actions"
WANTED = ["metadata_json", "narration_atomic_action_csv"]

# Feature lists copied from the notebook
FEATURES_V2 = [
    "head_trajectory_m", "head_duration_sec",
    "left_wrist_trajectory_m", "left_wrist_duration_sec",
    "right_wrist_trajectory_m", "right_wrist_duration_sec",
    "observer_trajectory_m", "observer_duration_sec",
    "num_actions", "unique_actions",
    "avg_action_duration", "median_action_duration",
    "avg_action_length", "max_action_length",
    "text_len_words", "text_unique_words", "text_diversity", "avg_word_length",
    "efficiency_score", "fragmentation", "motion_intensity",
    "avg_words_per_action", "wrist_motion_total", "wrist_motion_intensity",
    "participant_height_cm", "participant_weight_kg", "participant_bmi",
]
FEATURES_V2_FINAL = [  # the 18 features used for the original 0.989 result
    "head_trajectory_m", "head_duration_sec",
    "left_wrist_trajectory_m", "left_wrist_duration_sec",
    "right_wrist_duration_sec", "observer_duration_sec",
    "num_actions", "unique_actions",
    "avg_action_duration", "median_action_duration",
    "avg_action_length", "max_action_length",
    "text_len_words", "avg_word_length",
    "fragmentation", "participant_height_cm",
    "participant_weight_kg", "participant_bmi",
]
LABEL_INPUTS = [
    "efficiency_score", "fragmentation", "motion_intensity",
    "num_actions", "unique_actions", "head_trajectory_m", "action_duration_sec",
]


# ---------------------------------------------------------------- download
def download_one(url, path, expected_size):
    if os.path.exists(path) and os.path.getsize(path) == expected_size:
        return "already_exists"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.replace(tmp, path)
    return "downloaded"


def download_all(manifest_path, data_dir, workers=8):
    with open(manifest_path) as f:
        manifest = json.load(f)

    jobs = []
    for seq_id, seq in manifest["sequences"].items():
        for ftype in WANTED:
            if ftype in seq:
                e = seq[ftype]
                jobs.append((e["download_url"],
                             os.path.join(data_dir, seq_id, e["filename"]),
                             int(e["file_size_bytes"])))

    print(f"Files in manifest: {len(jobs)}")
    counts = {"already_exists": 0, "downloaded": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_one, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["failed"] += 1
                print("FAILED:", futures[fut][1], exc)
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)} {counts}")
    print("Download summary:", counts)
    if counts["failed"]:
        print("Some files failed. If many failed with 403 errors, the manifest links "
              "have expired: download a fresh manifest from the Project Aria site.")


# ---------------------------------------------------------------- features
def build_table(data_dir):
    meta_rows, atomic_frames = [], []
    for seq_id in sorted(os.listdir(data_dir)):
        seq_dir = os.path.join(data_dir, seq_id)
        if not os.path.isdir(seq_dir):
            continue
        for fname in os.listdir(seq_dir):
            fpath = os.path.join(seq_dir, fname)
            if fname.endswith("_metadata.json"):
                with open(fpath) as f:
                    rec = json.load(f)
                rec["sequence_id"] = seq_id
                meta_rows.append(rec)
            elif fname.endswith("_narration_atomic_action.csv"):
                tmp = pd.read_csv(fpath)
                tmp["sequence_id"] = seq_id
                atomic_frames.append(tmp)

    df_meta = pd.DataFrame(meta_rows)
    df_atomic = pd.concat(atomic_frames, ignore_index=True)
    print(f"Metadata records: {df_meta.shape}  Atomic narration rows: {df_atomic.shape}")

    df_atomic["action_duration"] = df_atomic["end_time"] - df_atomic["start_time"]
    text = df_atomic[TEXT_COL].map(str)  # str() per value, so missing text becomes "nan" as in the notebook
    df_atomic["action_length"] = text.map(len)

    words = text.map(str.split)
    n_words = words.map(len)
    n_unique = words.apply(lambda w: len(set(w)))
    df_atomic["text_len_words"] = n_words
    df_atomic["text_unique_words"] = n_unique
    df_atomic["text_diversity"] = np.where(n_words > 0, n_unique / n_words.replace(0, 1), 0)
    df_atomic["avg_word_length"] = words.apply(lambda w: sum(len(x) for x in w) / len(w) if w else 0)

    agg = df_atomic.groupby("sequence_id").agg(
        num_actions=("request_id", "count"),
        unique_actions=(TEXT_COL, "nunique"),
        avg_action_duration=("action_duration", "mean"),
        median_action_duration=("action_duration", "median"),
        text_len_words=("text_len_words", "sum"),
        text_unique_words=("text_unique_words", "mean"),
        text_diversity=("text_diversity", "mean"),
        avg_word_length=("avg_word_length", "mean"),
        avg_action_length=("action_length", "mean"),
        max_action_length=("action_length", "max"),
    ).reset_index()

    df = df_meta.merge(agg, on="sequence_id", how="left")

    df["efficiency_score"] = df["num_actions"] / (df["action_duration_sec"] + 1)
    df["fragmentation"] = df["num_actions"] / (df["unique_actions"] + 1)
    df["motion_intensity"] = df["head_trajectory_m"] / (df["action_duration_sec"] + 1)
    df["avg_words_per_action"] = df["text_len_words"] / (df["num_actions"] + 1)
    df["wrist_motion_total"] = df["left_wrist_trajectory_m"].fillna(0) + df["right_wrist_trajectory_m"].fillna(0)
    df["wrist_motion_intensity"] = df["wrist_motion_total"] / (df["action_duration_sec"] + 1)

    # V2 proxy label, quantiles computed before dropping rows (as in the notebook)
    df["discomfort_proxy"] = (
        (df["efficiency_score"] > df["efficiency_score"].quantile(0.75)) |
        (df["fragmentation"] > df["fragmentation"].quantile(0.75)) |
        (df["motion_intensity"] > df["motion_intensity"].quantile(0.75))
    ).astype(int)

    df = df.dropna(subset=FEATURES_V2 + ["discomfort_proxy"]).copy()
    df["participant_id"] = df["sequence_id"].str.extract(r"_s\d+_(.+?)_act\d+", expand=False)
    print(f"Modeling table: {df.shape}  participants: {df['participant_id'].nunique()}  "
          f"positive rate: {df['discomfort_proxy'].mean():.3f}")
    return df


# ---------------------------------------------------------------- modeling
def lgbm():
    return lgb.LGBMClassifier(num_leaves=20, max_depth=4, learning_rate=0.05, n_estimators=100,
                              feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1,
                              random_state=SEED, verbose=-1)


def models():
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced", random_state=SEED, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(max_depth=4, learning_rate=0.05, n_estimators=100, subsample=0.9,
                                     colsample_bytree=0.7, eval_metric="logloss",
                                     random_state=SEED, verbosity=0),
        "LightGBM": lgbm(),
    }


def cv_scores(model, X, y, groups=None):
    cv = (StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED) if groups is not None
          else StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED))
    s = cross_validate(model, X, y, groups=groups, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    return {"accuracy": s["test_accuracy"].mean(), "f1": s["test_f1"].mean(),
            "roc_auc": s["test_roc_auc"].mean(), "roc_auc_std": s["test_roc_auc"].std()}


def clean_features(df):
    derived = ["avg_words_per_action", "wrist_motion_intensity"]  # computed from label inputs
    print("Removed (computed from label inputs):", derived)
    candidates = [f for f in FEATURES_V2 if f not in LABEL_INPUTS + derived]
    corr = df[candidates + LABEL_INPUTS].corr().abs()
    leaky = [f for f in candidates if (corr.loc[f, LABEL_INPUTS] > 0.9).any()]
    keep = [f for f in candidates if f not in leaky]
    print("Removed label inputs:", [c for c in LABEL_INPUTS if c in FEATURES_V2])
    print("Removed (>0.9 corr with label inputs):", leaky)
    print(f"Clean features ({len(keep)}):", keep)
    return keep


def to_md(df):
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="Nymeria_download_urls.json")
    ap.add_argument("--data-dir", default="data/nymeria")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    if not args.skip_download:
        download_all(args.manifest, args.data_dir)

    df = build_table(args.data_dir)
    y, groups = df["discomfort_proxy"], df["participant_id"]
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Reproduce the original V2 setup: 18 features, random 80/10/10 split
    print("\n=== 1. Original V2 setup (random split, label-derived features) ===")
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(df[FEATURES_V2_FINAL], y, test_size=0.2,
                                                random_state=SEED, stratify=y)
    _, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=SEED, stratify=y_tmp)
    m = lgbm().fit(X_tr, y_tr)
    v2_auc = roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])
    print(f"LightGBM test ROC-AUC: {v2_auc:.3f}  (notebook reported 0.989)")

    # 2. Ablation: which fix matters?
    print("\n=== 2. Ablation (LightGBM, 5-fold CV) ===")
    clean = clean_features(df)
    ablation = pd.DataFrame([
        {"setup": "Original features, random CV", **cv_scores(lgbm(), df[FEATURES_V2_FINAL], y)},
        {"setup": "Original features, participant-grouped CV", **cv_scores(lgbm(), df[FEATURES_V2_FINAL], y, groups)},
        {"setup": "Leakage-free features, random CV", **cv_scores(lgbm(), df[clean], y)},
        {"setup": "Leakage-free features, participant-grouped CV", **cv_scores(lgbm(), df[clean], y, groups)},
    ]).set_index("setup").round(3)
    print(ablation.to_string())

    # 3. Corrected model comparison
    print("\n=== 3. Corrected comparison (leakage-free features, participant-grouped CV) ===")
    final = pd.DataFrame([{"model": n, **cv_scores(mdl, df[clean], y, groups)}
                          for n, mdl in models().items()]).set_index("model").round(3)
    print(final.to_string())

    ablation.to_csv(os.path.join(args.out_dir, "ablation.csv"))
    final.to_csv(os.path.join(args.out_dir, "corrected_results.csv"))
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write(f"Rows: {len(df)} | Participants: {groups.nunique()} | "
                f"Positive rate: {y.mean():.3f}\n\n")
        f.write(f"Original V2 setup, LightGBM test ROC-AUC: {v2_auc:.3f}\n\n")
        f.write("## Ablation (LightGBM)\n\n" + to_md(ablation) + "\n\n")
        f.write("## Corrected comparison\n\n" + to_md(final) + "\n")
    print(f"\nSaved results to {args.out_dir}/ (ablation.csv, corrected_results.csv, summary.md)")


if __name__ == "__main__":
    main()
