import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RQ1"))

import run_rq1_nli as rq1


class BackTranslationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "RQ1/configs/rq1_nli_config.json").open(encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_back_translation_is_presampled_and_seed_specific(self):
        args = Namespace(
            model="gemma-3-4b-it",
            experiment=["back_translation"],
            train_data=["snli"],
            seeds=[42, 43],
            target=None,
            mr_instruction_mode="pair_operation",
            lr=None,
            epochs=None,
            max_steps=None,
            rank=None,
            batch=None,
            grad_accum=None,
        )
        experiments, _ = rq1.build_experiment_list(self.config, args)
        self.assertEqual(len(experiments), 2)
        self.assertTrue(all(exp["skip_sample"] for exp in experiments))
        self.assertTrue(all(exp["mr_instruction_mode"] == "none" for exp in experiments))
        self.assertEqual(
            [exp["train_data"] for exp in experiments],
            [
                "data/nli/back_translation/snli_mettrain_gemma_v2_multipivot_5340_combined_seed42.jsonl",
                "data/nli/back_translation/snli_mettrain_gemma_v2_multipivot_5340_combined_seed43.jsonl",
            ],
        )

    def test_config_points_to_existing_5340_row_files(self):
        template = self.config["models"]["gemma-3-4b-it"]["training_data"]["snli"]["back_translation"]
        for seed in (42, 43, 44):
            path = ROOT / template.replace("{seed}", str(seed))
            self.assertTrue(path.is_file(), path)
            with path.open(encoding="utf-8") as handle:
                self.assertEqual(sum(1 for line in handle if line.strip()), 5340)


if __name__ == "__main__":
    unittest.main()
