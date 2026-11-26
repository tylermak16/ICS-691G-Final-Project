"""Reference readers and coordinate conventions for the GT donor experiment."""

import csv
import gzip
import hashlib
import io
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CHR21_LENGTH = 46_709_983
DNA = frozenset("ACGT")
COMPLEMENTS = str.maketrans("ACGTN", "TGCAN")
CHR21_ALIASES = frozenset({"chr21", "21", "NC_000021.9", "CM000683.2"})
CHR21_GTF_PREFIXES = tuple(f"{name}\t".encode("ascii") for name in sorted(CHR21_ALIASES))
ATTRIBUTES = re.compile(r'(\w+) "([^"]+)"')


def reverse_complement(sequence):
    return sequence.translate(COMPLEMENTS)[::-1]


def sequence_key(sequence):
    return min(sequence, reverse_complement(sequence))


def open_binary(path):
    path = Path(path)
    if path.suffix in {".gz", ".bgz", ".bgzf"}:
        return gzip.open(path, "rb")
    return path.open("rb")


def file_digest(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_chr21(path, check_eof=False):
    chunks = []
    header = None
    active = False
    with open_binary(path) as source:
        for line in source:
            if line.startswith(b">"):
                if active and not check_eof:
                    break
                name = line[1:].split(maxsplit=1)[0].decode("ascii")
                active = name in CHR21_ALIASES
                if active:
                    if header is not None:
                        raise ValueError("FASTA contains multiple chr21 records")
                    header = line[1:].decode("ascii").strip()
            elif active:
                chunks.append(line.strip().upper())
    if not chunks:
        raise ValueError("FASTA contains no nonempty GRCh38 chr21 record")
    return b"".join(chunks).decode("ascii"), header


@dataclass(frozen=True, order=True)
class Intron:
    gene: str
    transcript: str
    strand: str
    start: int
    end: int

    @property
    def donor(self):
        return self.start if self.strand == "+" else self.end - 1


def read_annotation(path, chr_length):
    transcripts = defaultdict(set)
    header = []
    counts = Counter()
    with open_binary(path) as source:
        for number, line in enumerate(source, 1):
            if line.startswith(b"#"):
                if not counts["records"]:
                    header.append(line.decode().strip())
                continue
            if not line.strip():
                continue
            counts["records"] += 1
            if not line.startswith(CHR21_GTF_PREFIXES):
                continue
            fields = line.decode().rstrip("\r\n").split("\t")
            if len(fields) != 9:
                raise ValueError(f"Malformed chr21 GTF record at line {number}")
            if fields[2] != "exon":
                continue
            start, end, strand = int(fields[3]) - 1, int(fields[4]), fields[6]
            if strand not in {"+", "-"} or not 0 <= start < end <= chr_length:
                raise ValueError(f"Invalid chr21 exon at GTF line {number}")
            attrs = dict(ATTRIBUTES.findall(fields[8]))
            key = (attrs["gene_id"], attrs["transcript_id"], strand)
            if (start, end) in transcripts[key]:
                counts["duplicate_exon_records"] += 1
            transcripts[key].add((start, end))
            counts["chr21_exon_records"] += 1

    if not transcripts:
        raise ValueError("GTF contains no chr21 exon records")
    introns = []
    for (gene, transcript, strand), exons in sorted(transcripts.items()):
        ordered = sorted(exons)
        for left, right in zip(ordered, ordered[1:]):
            start, end = left[1], right[0]
            if end <= start:
                counts["adjacent_or_overlapping_exons"] += 1
                continue
            if end - start == 1:
                counts["one_base_introns"] += 1
            introns.append(Intron(gene, transcript, strand, start, end))
    counts.update(chr21_transcripts=len(transcripts), chr21_intron_records=len(introns))
    return introns, header, counts


def motif_start(position, strand):
    return position if strand == "+" else position - 1


def window_bounds(position, strand, flank):
    start = motif_start(position, strand) - flank
    return start, start + 2 * flank + 2


def extract_window(chromosome, position, strand, flank):
    start, end = window_bounds(position, strand, flank)
    if not 0 <= start < end <= len(chromosome):
        return "", "window_outside_chromosome"
    sequence = chromosome[start:end]
    if not set(sequence) <= DNA:
        return "", "ambiguous_bases"
    if strand == "-":
        sequence = reverse_complement(sequence)
    return sequence, None


def dataset_summary(rows):
    return {
        split: dict(sorted(Counter(row["label"] for row in rows if row["split"] == split).items()))
        for split in ("train", "test")
    }


def make_row(sequence, label, position, intron, flank, pair_id):
    start, end = window_bounds(position, intron.strand, flank)
    return {
        "example_id": f"{pair_id}_{label}", "sequence": sequence, "label": label,
        "chromosome": "chr21", "position_0based": position, "strand": intron.strand,
        "gene_id": intron.gene, "transcript_id": intron.transcript,
        "intron_start_0based": intron.start, "intron_end_0based_exclusive": intron.end,
        "window_start_0based": start, "window_end_0based_exclusive": end,
        "pair_id": pair_id, "split": "",
    }


def candidate_positions(chromosome, intron, flank):
    reverse = intron.strand == "-"
    motif = "AC" if reverse else "GT"
    # Flanking bases may extend beyond the parent intron, as with the positive window.
    start = max(intron.start, flank)
    end = min(intron.end, len(chromosome) - flank)
    if end <= start:
        return []
    return [start + hit.start() + int(reverse) for hit in re.finditer(motif, chromosome[start:end])]


def build_pairs(chromosome, introns, pairs, flank, rng):
    annotations = {(intron.donor, intron.strand) for intron in introns}
    variants = defaultdict(list)
    seen_parents = set()
    for intron in sorted(introns):
        parent = (intron.gene, intron.strand, intron.start, intron.end)
        if parent not in seen_parents:
            variants[intron.donor, intron.strand].append(intron)
            seen_parents.add(parent)

    rows = []
    discards = Counter()
    candidate_discards = Counter()
    used_sequences = set()
    used_positions = set()
    donors = sorted(variants)
    rng.shuffle(donors)
    considered = 0
    for donor_position, strand in donors:
        if len(rows) == 2 * pairs:
            break
        considered += 1
        positive, error = extract_window(chromosome, donor_position, strand, flank)
        if error:
            discards[f"donor_{error}"] += 1
            continue
        if positive[flank:flank + 2] != "GT":
            discards["non_gt_annotated_donor"] += 1
            continue
        pos_key = sequence_key(positive)
        if pos_key in used_sequences:
            discards["duplicate_donor_sequence"] += 1
            continue
        parents = variants[donor_position, strand]
        rng.shuffle(parents)
        match = None
        for intron in parents:
            if intron.end - intron.start < 2:
                candidate_discards["parent_too_short_for_dinucleotide"] += 1
                continue
            positions = candidate_positions(chromosome, intron, flank)
            rng.shuffle(positions)
            for position in positions:
                if (position, strand) in annotations:
                    candidate_discards["annotated_donor_across_transcripts"] += 1
                    continue
                if (position, strand) in used_positions:
                    candidate_discards["reused_decoy_position"] += 1
                    continue
                decoy, error = extract_window(chromosome, position, strand, flank)
                if error:
                    candidate_discards[error] += 1
                    continue
                key = sequence_key(decoy)
                if key == pos_key or key in used_sequences:
                    candidate_discards["duplicate_decoy_sequence"] += 1
                    continue
                match = (intron, position, decoy, key)
                break
            if match is not None:
                break
        if match is None:
            discards["no_eligible_same_intron_decoy"] += 1
            continue
        intron, position, decoy, key = match
        pair_id = f"pair_{len(rows) // 2 + 1:05d}"
        rows.extend([
            make_row(positive, 1, donor_position, intron, flank, pair_id),
            make_row(decoy, 0, position, intron, flank, pair_id),
        ])
        used_sequences.update((pos_key, key))
        used_positions.add((position, strand))

    return rows, {
        "annotated_donors_considered": considered, "annotated_donors_available": len(donors),
        "discarded_donors": dict(discards), "rejected_decoy_candidates": dict(candidate_discards),
    }


def assign_splits(rows, test_fraction, rng):
    parents = {row["gene_id"]: row["gene_id"] for row in rows}

    def root(gene):
        while parents[gene] != gene:
            parents[gene] = parents[parents[gene]]
            gene = parents[gene]
        return gene

    # A connected run of overlapping windows has to stay in one split, even
    # when its windows belong to different genes or are on opposite strands.
    max_end = -1
    covering_gene = None
    ordered = sorted(rows, key=lambda row: row["window_start_0based"])
    for row in ordered:
        gene = row["gene_id"]
        if row["window_start_0based"] < max_end:
            parents[root(gene)] = root(covering_gene)
        if row["window_end_0based_exclusive"] > max_end:
            max_end = row["window_end_0based_exclusive"]
            covering_gene = gene

    groups = Counter(root(row["gene_id"]) for row in rows if row["label"] == 1)
    if len(groups) < 2:
        raise ValueError("Need at least two nonoverlapping gene groups for a held-out test split")
    pair_count = len(rows) // 2
    target = pair_count * test_fraction
    group_order = sorted(groups)
    best_gap = float("inf")
    test_roots = set()
    for _ in range(128):
        rng.shuffle(group_order)
        total = 0
        for index, group in enumerate(group_order[:-1], start=1):
            total += groups[group]
            gap = abs(total - target)
            if gap < best_gap:
                best_gap = gap
                test_roots = set(group_order[:index])
            if total >= target:
                break
        if best_gap == 0:
            break

    for row in rows:
        row["split"] = "test" if root(row["gene_id"]) in test_roots else "train"
    return len(groups), max(groups.values())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(rows, chromosome, introns, flank):
    transcript_introns = set(introns)
    donor_sites = {(intron.donor, intron.strand) for intron in transcript_introns}
    pairs = defaultdict(list)
    windows = []
    genes = {}
    seen_ids, seen_positions, seen_sequences = set(), set(), set()
    for row in rows:
        identifier = row["example_id"]
        strand, split, label = row["strand"], row["split"], row["label"]
        position = int(row["position_0based"])
        sequence = row["sequence"]
        require(row["chromosome"] == "chr21" and strand in {"+", "-"}, f"Invalid coordinate {identifier}")
        require(label in {"0", "1"} and split in {"train", "test"}, f"Invalid label or split {identifier}")
        require(len(sequence) == 2 * flank + 2 and set(sequence) <= DNA, f"Invalid sequence {identifier}")
        require(sequence[flank:flank + 2] == "GT", f"Central motif is not GT in {identifier}")
        assert_position = (position, strand)
        require(identifier not in seen_ids and assert_position not in seen_positions, f"Duplicate example {identifier}")
        key = sequence_key(sequence)
        require(key not in seen_sequences, f"Duplicate sequence or reverse complement {identifier}")
        seen_ids.add(identifier)
        seen_positions.add(assert_position)
        seen_sequences.add(key)

        start, end = window_bounds(position, strand, flank)
        require(start == int(row["window_start_0based"]) and end == int(row["window_end_0based_exclusive"]),
                f"Invalid window bounds {identifier}")
        require(extract_window(chromosome, position, strand, flank) == (sequence, None),
                f"Sequence disagrees with the reference {identifier}")
        parent = Intron(row["gene_id"], row["transcript_id"], strand,
                        int(row["intron_start_0based"]), int(row["intron_end_0based_exclusive"]))
        require(parent in transcript_introns, f"Parent intron missing from the annotation {identifier}")
        motif = motif_start(position, strand)
        require(parent.start <= motif and motif + 2 <= parent.end, f"Motif outside parent intron {identifier}")
        if label == "1":
            require(position == parent.donor, f"Positive is not the parent donor {identifier}")
        else:
            require(assert_position not in donor_sites, f"Decoy is an annotated donor {identifier}")
        require(genes.setdefault(parent.gene, split) == split, f"Gene leakage {parent.gene}")
        pairs[row["pair_id"]].append((label, split, parent))
        windows.append((start, end, split, identifier))

    for pair, members in pairs.items():
        require(sorted(item[0] for item in members) == ["0", "1"], f"Unbalanced pair {pair}")
        require(members[0][1:] == members[1][1:], f"Mismatched intron or split for {pair}")
    ends = {"train": -1, "test": -1}
    for start, end, split, identifier in sorted(windows):
        other = "test" if split == "train" else "train"
        require(start >= ends[other], f"Overlapping windows across splits near {identifier}")
        ends[split] = max(ends[split], end)
    summary = dataset_summary(rows)
    for split, counts in summary.items():
        require(counts.get("1", 0) == counts.get("0", 0) > 0, f"Missing or unbalanced {split} split")
    return {"pairs": len(pairs), "split_class_counts": summary, "genes": len(genes)}


def dataset_tsv(rows):
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return text.getvalue()


def prepare_dataset(fasta, gtf, n_pairs, flank, seed):
    chromosome, _ = read_chr21(fasta, check_eof=True)
    require(len(chromosome) == CHR21_LENGTH, f"Expected GRCh38 chr21 length {CHR21_LENGTH}")
    introns, header, gtf_counts = read_annotation(gtf, len(chromosome))
    description = " ".join(header).lower()
    require("grch38" in description and "gencode" in description,
            "GTF header does not identify a GENCODE GRCh38 annotation")
    rng = random.Random(seed)
    rows, sampling = build_pairs(chromosome, introns, n_pairs, flank, rng)
    require(len(rows) == 2 * n_pairs, f"Only found {len(rows) // 2} of {n_pairs} pairs: {sampling}")
    components, _ = assign_splits(rows, 0.2, rng)
    labels_as_text = [{**row, "label": str(row["label"])} for row in rows]
    validation = validate(labels_as_text, chromosome, introns, flank)
    input_files = {
        kind: {"file": path.name, "bytes": path.stat().st_size, "sha256": file_digest(path)}
        for kind, path in [("fasta", fasta), ("gtf", gtf)]
    }
    return rows, {
        **sampling, **validation, "gtf_counts": gtf_counts, "gene_overlap_components": components,
        "input_files": input_files, "dataset_sha256": hashlib.sha256(dataset_tsv(rows).encode()).hexdigest(),
    }
