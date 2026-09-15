import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RQ1"))

import run_rq1_nli as rq1


class GenericLLMAugConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "RQ1/configs/rq1_nli_config.json").open(encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def _args(self, seeds=(42, 43, 44)):
        return Namespace(
            model="gemma-3-4b-it",
            experiment=["generic_llm_aug"],
            train_data=["snli"],
            seeds=list(seeds),
            target=None,
            mr_instruction_mode="pair_operation",
            lr=None,
            epochs=None,
            max_steps=None,
            rank=None,
            batch=None,
            grad_accum=None,
        )

    def test_is_a_presampled_plain_nli_comparator(self):
        experiments, _ = rq1.build_experiment_list(self.config, self._args())
        self.assertEqual(
            [experiment["name"] for experiment in experiments],
            [
                "rq1_generic_llm_aug_snli_seed42",
                "rq1_generic_llm_aug_snli_seed43",
                "rq1_generic_llm_aug_snli_seed44",
            ],
        )
        self.assertTrue(all(experiment["skip_sample"] for experiment in experiments))
        self.assertTrue(all(experiment["mr_instruction_mode"] == "none" for experiment in experiments))
        self.assertTrue(all(experiment["target"] == 5340 for experiment in experiments))
        self.assertTrue(all("/other/" not in f"/{experiment['train_data']}" for experiment in experiments))

    def test_seed_files_are_exactly_5340_rows_and_well_formed(self):
        template = self.config["models"]["gemma-3-4b-it"]["training_data"]["snli"]["generic_llm_aug"]
        for seed in (42, 43, 44):
            path = ROOT / template.replace("{seed}", str(seed))
            self.assertTrue(path.is_file(), path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows), 5340)
            self.assertEqual(len({str(row["pair_id"]) for row in rows}), 5340)
            self.assertTrue(all(row.get("mr_id") == "none" for row in rows))
            self.assertTrue(all(row.get("premise", "").strip() and row.get("hypothesis", "").strip() for row in rows))


if __name__ == "__main__":
    unittest.main()
