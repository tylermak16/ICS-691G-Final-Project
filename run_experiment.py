"""Run the chromosome 21 baseline and frozen Caduceus layer sweep."""

import time

STARTED = time.perf_counter()

import argparse
import io
import json
import resource
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from caduceus import CONFIG, aligned_input_ids, gpu_memory, hidden_states, load_model
from dataset import dataset_tsv, prepare_dataset


METRICS = {"accuracy": "Accuracy", "auroc": "AUROC", "average_precision": "AUPRC (average precision)"}


@torch.inference_mode()
def extract_layers(model, tokenizer, sequences, flank, batch_size):
    features = np.empty((model.config.n_layer + 1, len(sequences), model.config.d_model), dtype=np.float32)
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start:start + batch_size]
        ids = aligned_input_ids(tokenizer, batch, flank)
        states = hidden_states(model, ids)
        selected = torch.stack([state[:, flank, :] for state in states])
        features[:, start:start + len(batch)] = selected.cpu().numpy()
    return features


def score(features, labels, train, seed):
    if not np.isfinite(features).all():
        raise ValueError("Non-finite classifier features")
    classifier = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    )
    classifier.fit(features[train], labels[train])
    scores = classifier.predict_proba(features[~train])[:, 1]
    return {
        "accuracy": accuracy_score(labels[~train], classifier.predict(features[~train])),
        "auroc": roc_auc_score(labels[~train], scores),
        "average_precision": average_precision_score(labels[~train], scores),
    }


def plot_layers(baseline, layers, n_test, length):
    figure, axes = plt.subplots(1, 3, sharey=True, figsize=(10, 4.3), layout="constrained")
    x = [record["layer"] for record in layers]
    minimum = min(0.45, min(record[key] for record in layers for key in METRICS) - 0.035)
    for ax, (key, title) in zip(axes, METRICS.items()):
        ax.plot(x, [record[key] for record in layers], color="#08796e", marker="o", markersize=4,
                linewidth=1.8, label="Caduceus probe")
        ax.axhline(baseline[key], color="#3f5268", linestyle="--", linewidth=1.5, label="One-hot baseline")
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks(range(0, x[-1] + 1, 2))
        ax.set_ylim(minimum, 1)
        ax.grid(axis="y", color="#dce1e6", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_xlabel("Caduceus layer")
    axes[0].set_ylabel("Held-out score")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    figure.suptitle(f"Chromosome 21 GT donor classification | {n_test} held-out sequences, {length} bases each",
                    fontsize=13, fontweight="bold")
    image = io.BytesIO()
    figure.savefig(image, dpi=180, format="png")
    plt.close(figure)
    return image.getvalue()


def run(args):
    seconds = {"imports_and_setup": time.perf_counter() - STARTED}
    stage = time.perf_counter()
    rows, dataset_info = prepare_dataset(args.fasta, args.gtf, args.pairs, args.flank, args.seed)
    dataset = dataset_tsv(rows)
    sequences = [row["sequence"] for row in rows]
    labels = np.array([int(row["label"]) for row in rows])
    train = np.array([row["split"] == "train" for row in rows])
    seconds["dataset_build_and_validation"] = time.perf_counter() - stage

    stage = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    tokenizer, model, _ = load_model()
    torch.cuda.synchronize()
    seconds["model_load"] = time.perf_counter() - stage

    stage = time.perf_counter()
    features = extract_layers(model, tokenizer, sequences, args.flank, args.batch_size)
    torch.cuda.synchronize()
    seconds["all_layer_embedding_extraction"] = time.perf_counter() - stage

    stage = time.perf_counter()
    with threadpool_limits(limits=1):
        bases = np.array([list(sequence) for sequence in sequences])
        one_hot = (bases[:, :, None] == np.array(list("ACGT"))).reshape(len(rows), -1).astype(np.float64)
        baseline = score(one_hot, labels, train, args.seed)
        layers = [
            {"layer": layer, **score(vectors, labels, train, args.seed)}
            for layer, vectors in enumerate(features)
        ]
    seconds["baseline_and_probe_fit_evaluation"] = time.perf_counter() - stage

    stage = time.perf_counter()
    n_test = int((~train).sum())
    length = len(sequences[0])
    figure = plot_layers(baseline, layers, n_test, length)
    metrics = {"seed": args.seed, "model": CONFIG["model_id"], "revision": CONFIG["revision"],
               "baseline": baseline, "layers": layers}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "dataset.tsv").write_text(dataset, encoding="utf-8", newline="\n")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (args.out / "layers.png").write_bytes(figure)
    seconds["plot_and_result_writes"] = time.perf_counter() - stage
    seconds["total"] = time.perf_counter() - STARTED
    benchmark = {
        "input_files": dataset_info.pop("input_files"),
        "examples": len(rows), "bases_per_example": length, "total_input_bases": len(rows) * length,
        "train_examples": int(train.sum()), "test_examples": n_test, "seed": args.seed,
        "batch_size": args.batch_size, "embedding_shape": list(features.shape),
        "embedding_array_bytes": features.nbytes, "dataset": dataset_info,
        "runtime_seconds": seconds,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "peak_gpu_memory": gpu_memory(), "gpu": torch.cuda.get_device_name(),
        "python": sys.version.split()[0], "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
    }
    report = json.dumps(benchmark, indent=2) + "\n"
    (args.out / "benchmark.json").write_text(report, encoding="utf-8")
    print(report, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True, help="GRCh38 genome or chromosome 21 FASTA")
    parser.add_argument("--gtf", type=Path, required=True, help="Comprehensive GENCODE GRCh38 annotation")
    parser.add_argument("--out", type=Path, default=Path("results/pilot"))
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--flank", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.pairs < 2 or args.flank < 1 or args.batch_size < 1:
        parser.error("Use at least two pairs, one flanking base and a positive batch size.")
    if not torch.cuda.is_available():
        parser.error("The pinned model requires the CUDA-enabled Linux environment.")
    run(args)


if __name__ == "__main__":
    main()
