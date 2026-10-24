"""Compare window-random, source-file and speed holdouts without exporting models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config import F_ROT, FS, GMF, N_POINTS, RANDOM_SEED, SEU_GEAR_CHANNEL, SEGMENT_OVERLAP
from features import extract_batch
from load_seu import detect_class, detect_speed, load_file


def group_arrays(y, groups):
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups) or not len(y):
        raise ValueError("Class labels and source groups must be nonempty aligned one-dimensional arrays")
    return y, groups


def file_holdout(y, groups, seed=RANDOM_SEED):
    """Keep a complete source file per class in test; never fall back to window CV."""
    y, groups = group_arrays(y, groups)
    rng = np.random.default_rng(seed)
    test_groups = []
    for group in np.unique(groups):
        if len(np.unique(y[groups == group])) != 1:
            raise ValueError("Each source file must have one class")
    for label in np.unique(y):
        candidates = np.unique(groups[y == label])
        if len(candidates) < 2:
            raise ValueError(f"Class {label!r} needs at least two independent source files")
        test_groups.append(rng.choice(candidates))
    test = np.flatnonzero(np.isin(groups, test_groups))
    train = np.flatnonzero(~np.isin(groups, test_groups))
    return train, test


def speed_holdouts(y, groups, speeds):
    y, groups = group_arrays(y, groups)
    speeds = np.asarray(speeds, dtype=float)
    if speeds.ndim != 1 or len(speeds) != len(y):
        raise ValueError("Operating speeds must align with class labels")
    if not np.isfinite(speeds).all() or (speeds <= 0).any() or len(np.unique(speeds)) < 2:
        raise ValueError("All files need known speeds and at least two regimes")
    for speed in np.unique(speeds):
        train, test = np.flatnonzero(speeds != speed), np.flatnonzero(speeds == speed)
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Source file crosses operating regimes")
        if set(y[test]) != set(y[train]):
            raise ValueError("Every speed holdout requires the same class coverage in train and test")
        yield str(int(speed)), train, test


def load_records(directory: Path, max_windows: int):
    signals, labels, groups, speeds, manifest = [], [], [], [], []
    source_hashes = {}
    files = sorted(path for path in directory.rglob("*") if path.suffix.lower() in {".csv", ".txt", ".dat"})
    step = max(1, int(N_POINTS * (1 - SEGMENT_OVERLAP)))
    for path in files:
        label, speed = detect_class(path.name), detect_speed(path.name)
        if label is None:
            continue
        if speed is None:
            raise ValueError(f"Unknown speed for {path.name}; refusing an assumed regime")
        data = load_file(str(path))["data"]
        channel = SEU_GEAR_CHANNEL if data.shape[1] > SEU_GEAR_CHANNEL else 0
        signal = data[:, channel]
        starts = list(range(0, len(signal) - N_POINTS + 1, step))
        if max_windows and len(starts) > max_windows:
            starts = [starts[index] for index in np.linspace(0, len(starts) - 1, max_windows, dtype=int)]
        if not starts:
            raise ValueError(f"Recording too short: {path}")
        group = path.relative_to(directory).as_posix()
        signals.extend(signal[start:start + N_POINTS] for start in starts)
        labels.extend([label] * len(starts))
        groups.extend([group] * len(starts))
        speeds.extend([speed] * len(starts))
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        fingerprint = digest.hexdigest()
        if fingerprint in source_hashes:
            raise ValueError(f"Duplicate recording contents: {group} and {source_hashes[fingerprint]}")
        source_hashes[fingerprint] = group
        manifest.append({"path": group, "sha256": fingerprint, "windows": len(starts),
                         "speed": speed, "class": label, "selected_channel_zero_based": channel})
    if not signals:
        raise ValueError("No recognized SEU recordings found")
    X, _ = extract_batch(np.asarray(signals))
    if not np.isfinite(X).all():
        raise ValueError("Feature extraction returned nonfinite values")
    return X, np.asarray(labels), np.asarray(groups), np.asarray(speeds), manifest


def score_split(X, y, groups, train, test, *, allow_source_overlap=False):
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    train, test = np.asarray(train), np.asarray(test)
    if X.ndim != 2 or len(X) != len(y) or len(y) != len(groups) or not np.isfinite(X).all():
        raise ValueError("Expected aligned finite feature matrix, labels and source groups")
    if not len(train) or not len(test):
        raise ValueError("Train and test must both be nonempty")
    if set(train) & set(test):
        raise ValueError("Train and test row overlap")
    if not allow_source_overlap and set(groups[train]) & set(groups[test]):
        raise ValueError("Source file crosses train/test; overlap is allowed only for the diagnostic split")
    if set(y[train]) != set(y[test]):
        raise ValueError("Train and test require the same class coverage")
    model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=4)
    model.fit(X[train], y[train])
    predicted = model.predict(X[test])
    classes = sorted(set(y))
    return {"accuracy": float(accuracy_score(y[test], predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], predicted)),
            "macro_f1": float(f1_score(y[test], predicted, labels=classes, average="macro", zero_division=0)),
            "classes": classes, "confusion_matrix": confusion_matrix(y[test], predicted, labels=classes).tolist(),
            "train_windows": len(train), "test_windows": len(test),
            "train_sources": sorted(set(groups[train])), "test_sources": sorted(set(groups[test])),
            "source_overlap": sorted(set(groups[train]) & set(groups[test]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/validation.json"))
    parser.add_argument("--max-windows-per-file", type=int, default=100)
    args = parser.parse_args()
    if args.max_windows_per_file < 0:
        parser.error("max-windows-per-file must be nonnegative")
    X, y, groups, speeds, manifest = load_records(args.data_dir, args.max_windows_per_file)
    indices = np.arange(len(y))
    train, test = train_test_split(indices, test_size=.2, stratify=y, random_state=RANDOM_SEED)
    results = {"window_random_diagnostic_only": score_split(X, y, groups, train, test, allow_source_overlap=True)}
    train, test = file_holdout(y, groups)
    results["source_file_holdout"] = score_split(X, y, groups, train, test)
    for speed, train, test in speed_holdouts(y, groups, speeds):
        results[f"heldout_speed_{speed}"] = score_split(X, y, groups, train, test)
    report = {"status": "completed", "seed": RANDOM_SEED, "sklearn": sklearn.__version__,
              "data_manifest": manifest, "max_windows_per_file": args.max_windows_per_file,
              "segmentation": {"points": N_POINTS, "overlap": SEGMENT_OVERLAP}, "results": results,
              "feature_reference": {"sampling_hz": FS, "rotation_hz": F_ROT, "gear_mesh_hz": GMF,
                                    "policy": "fixed feature basis for every recording and split; no speed-specific adaptation"},
              "interpretation": "Fixed baseline; no model selection using these test outcomes. Random-window is descriptive only.",
              "limitations": ["Source file identity does not prove independence of physical bearing, session or material.",
                              "Few files/regimes cannot establish cross-machine or field performance.",
                              "Window counts are not counts of independent physical samples."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value["macro_f1"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
