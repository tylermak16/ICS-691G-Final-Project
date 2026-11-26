import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import assign_splits, build_pairs, make_row
from dataset import Intron, extract_window, read_annotation, reverse_complement, window_bounds
from dataset import validate


class CoordinateTests(unittest.TestCase):
    def test_gtf_endpoints_and_transcription_orientation(self):
        with tempfile.TemporaryDirectory() as directory:
            gtf = Path(directory) / "example.gtf"
            lines = []
            for strand, gene, transcript, start, stop in [
                ("+", "g1", "t1", 131, 140), ("+", "g1", "t1", 101, 110),
                ("-", "g2", "t2", 221, 230), ("-", "g2", "t2", 201, 210),
                ("+", "g1", "t1", 131, 140), ("+", "g1", "t3", 151, 160),
                ("+", "g1", "t3", 101, 110),
            ]:
                lines.append(f'chr21\ttest\texon\t{start}\t{stop}\t.\t{strand}\t.\tgene_id "{gene}"; transcript_id "{transcript}";')
            gtf.write_text("\n".join(lines) + "\n", encoding="ascii")
            introns, _, counts = read_annotation(gtf, 300)
        self.assertEqual(introns, [Intron("g1", "t1", "+", 110, 130), Intron("g1", "t3", "+", 110, 150), Intron("g2", "t2", "-", 210, 220)])
        self.assertEqual([intron.donor for intron in introns], [110, 110, 219])
        self.assertEqual(counts["duplicate_exon_records"], 1)

    def test_strand_specific_windows(self):
        self.assertEqual(window_bounds(4, "+", 2), (2, 8))
        self.assertEqual(extract_window("AACCGTAAGCA", 4, "+", 2), ("CCGTAA", None))
        self.assertEqual(window_bounds(5, "-", 2), (2, 8))
        self.assertEqual(extract_window("TTGGACAAGGAA", 5, "-", 2), ("TTGTCC", None))
        self.assertEqual(reverse_complement("ACCTGN"), "NCAGGT")
        self.assertEqual(extract_window("NACCGTA", 2, "+", 2)[1], "ambiguous_bases")
        self.assertEqual(extract_window("ACCGTA", 1, "+", 2)[1], "window_outside_chromosome")


class DatasetTests(unittest.TestCase):
    def setUp(self):
        chromosome = list(random.Random(12).choices("ACGT", k=1500))
        chromosome[90:92] = "GT"
        chromosome[508:510] = "AC"
        chromosome[760:762] = "GT"
        self.chromosome = "".join(chromosome)
        self.introns = [Intron("g1", "t1", "+", 90, 190), Intron("g2", "t2", "-", 410, 510), Intron("g3", "t3", "+", 760, 860)]
        rows, _ = build_pairs(self.chromosome, self.introns, 3, 10, random.Random(42))
        assign_splits(rows, 1 / 3, random.Random(42))
        self.rows = [{key: str(value) for key, value in row.items()} for row in rows]

    def test_balanced_pairs_and_reference_agreement(self):
        report = validate(self.rows, self.chromosome, self.introns, 10)
        self.assertEqual(report["pairs"], 3)
        self.assertEqual(report["split_class_counts"], {"train": {"0": 2, "1": 2}, "test": {"0": 1, "1": 1}})

    def test_decoys_exclude_donors_from_another_transcript(self):
        chromosome = list("A" * 70)
        for position, sequence in [(9, "CACGT"), (21, "CCAGT"), (33, "ACTGT")]:
            chromosome[position:position + len(sequence)] = sequence
        parent = Intron("g1", "t1", "+", 12, 44)
        other = Intron("g1", "t2", "+", 24, 44)
        rows, report = build_pairs("".join(chromosome), [parent, other], 2, 3, random.Random(42))
        self.assertEqual([row["position_0based"] for row in rows if row["label"] == 0], [36])
        self.assertGreater(report["rejected_decoy_candidates"]["annotated_donor_across_transcripts"], 0)

    def test_validator_rejects_motif_change_duplicate_and_unbalanced_pair(self):
        altered = [dict(row) for row in self.rows]
        altered[0]["sequence"] = "A" * 22
        with self.assertRaisesRegex(ValueError, "Central motif"):
            validate(altered, self.chromosome, self.introns, 10)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate(self.rows + [self.rows[0]], self.chromosome, self.introns, 10)
        with self.assertRaisesRegex(ValueError, "Unbalanced pair"):
            validate(self.rows[1:], self.chromosome, self.introns, 10)

    def test_validator_rejects_annotated_decoy(self):
        decoy = next(row for row in self.rows if row["label"] == "0")
        position = int(decoy["position_0based"])
        reverse = decoy["strand"] == "-"
        extra = Intron("alternate_gene", "alternate_isoform", decoy["strand"],
                       position - 40 if reverse else position, position + 1 if reverse else position + 40)
        with self.assertRaisesRegex(ValueError, "Decoy is an annotated donor"):
            validate(self.rows, self.chromosome, [*self.introns, extra], 10)

    def test_gene_leakage_is_rejected(self):
        self.rows[0]["split"] = "test" if self.rows[0]["split"] == "train" else "train"
        with self.assertRaisesRegex(ValueError, "Gene leakage"):
            validate(self.rows, self.chromosome, self.introns, 10)

    def test_overlapping_windows_from_different_genes_stay_together(self):
        chromosome = list(self.chromosome)
        for position in (99, 130, 160, 800):
            chromosome[position:position + 2] = "GT"
        chromosome = "".join(chromosome)
        group = [self.introns[0], Intron("overlap", "tx_overlap", "+", 99, 250), self.introns[2]]
        rows = []
        for index, (parent, decoy) in enumerate(zip(group, [130, 160, 800])):
            for label, position in [(1, parent.donor), (0, decoy)]:
                sequence, error = extract_window(chromosome, position, parent.strand, 10)
                self.assertIsNone(error)
                rows.append(make_row(sequence, label, position, parent, 10, f"pair_{index}"))
        self.assertEqual(assign_splits(rows, 1 / 3, random.Random(42)), (2, 2))
        self.assertEqual(rows[0]["split"], rows[2]["split"])
        self.assertNotEqual(rows[0]["split"], rows[4]["split"])
        for row in rows[2:4]:
            row["split"] = rows[4]["split"]
        with self.assertRaisesRegex(ValueError, "Overlapping windows across splits"):
            validate([{key: str(value) for key, value in row.items()} for row in rows], chromosome, group, 10)


if __name__ == "__main__":
    unittest.main()
