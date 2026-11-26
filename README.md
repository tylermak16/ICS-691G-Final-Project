# Probing Caduceus for splice-donor information

This prototype tests whether a linear classifier can distinguish annotated GT splice donors from matched unannotated GT sites using a frozen Caduceus representation. It compares the input embedding and all 16 Caduceus-Ph blocks with a position-specific one-hot baseline.

## Run the prototype

Open Ubuntu under WSL2 with an NVIDIA GPU. Use `bash setup_wsl.sh` to create the pinned Python 3.10 and CUDA 11.8 environment. The first model load downloads and verifies the files pinned in `model.json`. Run from the project folder.

```bash
cd /mnt/c/Users/tyler/OneDrive/Documents/ChatGPT/691G/splice_probing
bash setup_wsl.sh
python_bin="$HOME/.local/share/splice-probing/venv/bin/python"
"$python_bin" -B -m unittest discover -s tests -v
"$python_bin" -B run_experiment.py \
  --fasta data/raw/chr21.fa.gz \
  --gtf /mnt/c/Users/tyler/Downloads/gencode.v50.primary_assembly.annotation.gtf.gz \
  --out results/reproduction
```

The required inputs are the chromosome 21 GRCh38 FASTA and the comprehensive GENCODE v50 primary-assembly GTF. Both can be read as compressed files. Place the FASTA at `data/raw/chr21.fa.gz` and point `--gtf` to the downloaded GTF. Input file digests, sample counts, and the validated dataset fingerprint are recorded in the benchmark JSON. [UCSC GRCh38 FASTA](https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz) · [GENCODE v50](https://www.gencodegenes.org/human/release_50.html) · [Caduceus paper](https://arxiv.org/abs/2403.03234)

## Method

The pilot contains 500 annotated donors and 500 same-intron decoys from 271 genes. Both classes have a central GT and 100 bases on either side. A decoy cannot overlap an annotated donor at its motif position on the same strand. With seed 42, 800 examples are used for training and 200 for testing. The split keeps genes, matched pairs, and overlapping windows together. Validation checks labels against the FASTA and GTF and checks for duplicate windows and leakage.

The baseline fits logistic regression to the 808 position-specific one-hot features. Each Caduceus probe fits its own logistic regression to the 256-dimensional vector at the first G in the central motif. The input embedding is layer 0, and layer 16 is the final normalized representation. Scaling and classifier fitting use training examples only.

## Preliminary benchmark

One end-to-end run on the RTX 4060 Laptop GPU in WSL2, using the cached model and a batch size of 32.

| Measurement | Observed |
| --- | ---: |
| Compressed FASTA and GTF inputs | 131.0 MiB |
| Model input | 1,000 × 202 bases (202,000 bases total) |
| End-to-end runtime | 117.56 s |
| Dataset construction and validation | 95.80 s |
| All-layer vector extraction | 4.34 s |
| Peak Python process RAM (RSS) | 842.7 MiB |
| Peak PyTorch GPU memory | 357.8 MiB allocated, 544 MiB reserved |

The pilot covers one chromosome. The final sample count is open. A 10,000-window run would have ten times as many examples and would hold about 166 MiB of layer vectors in host RAM, versus 16.6 MiB here. Extraction time should grow with the number of batches. Peak GPU memory should be similar at the same window length and batch size. Longer windows can increase per-batch GPU memory, and expanding to additional chromosomes adds reference parsing.

## Preliminary results

The [layer figure](results/pilot/layers.png) plots accuracy, AUROC, and average precision against model depth. Layer 1 reached 0.970 AUROC, the highest observed on this split. The one-hot baseline reached 0.857 and layer 8 reached 0.758. Layers 1–3 and 15–16 exceeded baseline AUROC. These comparisons are preliminary and use a single held-out split. The decoys are annotation-defined presumed negatives, and differences in exon and intron composition could contribute to the scores. The authors' pretraining recipe uses the human reference genome. [Caduceus pretraining recipe](https://github.com/kuleshov-group/caduceus/tree/0060a6d8079b6a040fc55d505e15972a327b70a6#pretraining-on-human-reference-genome)

The run generates `dataset.tsv`, `benchmark.json`, `metrics.json`, and `layers.png`. The [benchmark record](results/pilot/benchmark.json) and [layer scores](results/pilot/metrics.json) contain the exact measured values. Raw references and the generated dataset stay out of Git.
