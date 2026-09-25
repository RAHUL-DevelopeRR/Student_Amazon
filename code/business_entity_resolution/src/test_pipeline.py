import unittest
import csv
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from metric import entity_f05, macro_f05
from normalize import normalize, numeric_tokens, expanded, accent_fold


class CoreTests(unittest.TestCase):
    def test_metric(self):
        self.assertEqual(entity_f05([], []), 1)
        self.assertEqual(entity_f05([], ["x"]), 0)
        self.assertEqual(entity_f05(["x"], []), 0)
        self.assertEqual(entity_f05(["x"], ["y"]), 0)
        self.assertEqual(entity_f05(["x"], ["x"]), 1)
        self.assertAlmostEqual(entity_f05(["a", "b"], ["a", "b", "c"]), 5 / 7)
        self.assertAlmostEqual(entity_f05(["a", "b"], ["a"]), 5 / 6)
        self.assertAlmostEqual(macro_f05({"1": set(), "2": {"a"}}, {"1": set(), "2": set()}), .5)
        with self.assertRaises(ValueError):
            macro_f05({"1": set()}, {})
        with self.assertRaises(ValueError):
            macro_f05({}, {})

    def test_normalization(self):
        self.assertEqual(normalize("  A&B, ＬＴＤ. "), "a and b ltd")
        self.assertEqual(expanded("ABC Pvt Ltd"), "abc private limited")
        self.assertEqual(normalize("École"), "école")
        self.assertEqual(accent_fold("École"), "ecole")
        self.assertEqual(numeric_tokens("0012 / 12B - 75001"), ("0012", "12", "75001"))
        self.assertEqual(normalize(None), "")

    def test_retrieval_shards_and_countries(self):
        from blocking import retrieve, metrics
        q = [("S1-1", "ecole paris", "12 rue paris", "France"),
             ("S1-2", "", "", "France")]
        corpus = [("S2-1", "ecole paris", "12 rue paris", "France"),
                  ("S2-2", "ecole paris", "12 rue paris", "US"),
                  ("S3-1", "ecole paris", "12 rue paris", ""),
                  ("S3-2", "bakery", "99 road", "France")]
        full, _, _ = retrieve(q, lambda: iter([corpus]), corpus, k=1)
        shards, _, _ = retrieve(q, lambda: iter([[r] for r in corpus]), corpus, k=1)
        self.assertEqual(full, shards)
        self.assertIn("S2-1", full[0])
        self.assertIn("S3-1", full[0])
        self.assertNotIn("S2-2", full[0])
        self.assertEqual(full[1], set())
        m = metrics(q, {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}, full, 4)
        self.assertEqual(m["candidate_recall"], 1)
        self.assertEqual(m["oracle_macro_f05_ceiling"], 1)

    def test_official_formula_exhaustive(self):
        # Independent precision/recall calculation for a grid of TP/FP/FN.
        for tp in range(5):
            for fp in range(5):
                for fn in range(5):
                    truth = set(range(tp + fn))
                    pred = set(range(tp)) | set(range(100,100+fp))
                    precision = tp/(tp+fp) if tp+fp else 0
                    recall = tp/(tp+fn) if tp+fn else 0
                    expected = (1.25*precision*recall/(.25*precision+recall)
                                if precision+recall else float(not truth and not pred))
                    self.assertAlmostEqual(entity_f05(truth,pred), expected)

    def test_complete_audit_on_fixture(self):
        import profile_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ["train", "test"]:
                (root / split).mkdir()
                for source in range(1,4):
                    with (root / split / f"{split}_source{source}.tsv").open("w",encoding="utf-8",newline="") as f:
                        w=csv.writer(f,delimiter="\t")
                        w.writerow(["entity_id","business_name","business_address","country"])
                        w.writerow([f"S{source}-1","École", "12, rue", "France"])
                        w.writerow([f"S{source}-2","", "", "US"])
            with (root / "train/train_ground_truth.tsv").open("w",encoding="utf-8",newline="") as f:
                w=csv.writer(f,delimiter="\t")
                w.writerow(["source1_entity_id","matched_entity_ids"])
                w.writerow(["S1-1","S2-1,S3-1"])
                w.writerow(["S1-2",""])
            with patch("sys.argv",["profile_data.py","--data-dir",str(root),"--artifacts",str(root / "artifacts")]), patch.object(profile_data,"log_experiment"):
                profile_data.main()
            report=json.loads((root / "artifacts/data_profile.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"],"complete")
            self.assertEqual(report["ground_truth"]["all_empty_macro_f05"],.5)
            self.assertEqual(report["ground_truth"]["unknown_target_ids"],0)
            self.assertEqual(report["files"]["test_source1"]["business_name"]["missing"],1)
            from config import connect
            from blocking import exact_diagnostic
            con=connect(root / "artifacts/audit.duckdb")
            con.execute("CREATE TEMP VIEW queries AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM train_source1")
            con.execute("CREATE TEMP VIEW corpus AS SELECT entity_id,name_norm,address_norm,coalesce(country,'') country FROM targets")
            queries=con.execute("SELECT * FROM queries ORDER BY entity_id").fetchall()
            candidates,_,_=exact_diagnostic(con,queries)
            self.assertEqual(candidates,[{"S2-1","S3-1"},set()])
            con.close()


if __name__ == "__main__":
    unittest.main()
