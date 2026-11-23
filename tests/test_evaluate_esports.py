import csv
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.data import clean, load_raw
from src.evaluate_esports import (
    SNAPSHOT_END, TEST_START, add_series_groups, main, paired_brier_bootstrap, validation_folds,
)


def match(game_id, date, result, team=None, opponent=None, league="LCK", patch_id="16.1"):
    return {
        "game_id": game_id, "date": date, "result": result, "side": "Blue",
        "team_name": team if team is not None else f"Team {game_id}",
        "opponent_team_name": opponent if opponent is not None else f"Opponent {game_id}",
        "league": league, "patch": patch_id, "goldat10": 15000,
        "golddiffat10": 2000 * result - 1000 + game_id * 20,
        "xpdiffat10": 1200 * result - 600, "csdiffat10": 20 * result - 10,
        "killsat10": 1 + result, "deathsat10": 2 - result,
    }


class EsportsEvaluationTests(unittest.TestCase):
    def test_series_identity_and_validation_boundaries(self):
        data = pd.DataFrame([
            match(2 * i + result, date, result)
            for i, date in enumerate(pd.date_range("2026-02-01", periods=12, freq="h"))
            for result in (0, 1)
        ])
        data.loc[3, ["team_name", "opponent_team_name"]] = ["A", "B"]
        data.loc[5, ["team_name", "opponent_team_name"]] = ["B", "A"]
        data.loc[6, ["league", "team_name", "opponent_team_name"]] = ["LPL", "A", "B"]
        data.loc[[7, 8], ["team_name", "opponent_team_name"]] = [pd.NA, "A"]
        grouped = add_series_groups(data)
        keys = grouped["series_group"]
        self.assertEqual(keys[3], keys[5])
        self.assertNotEqual(keys[3], keys[6])
        self.assertNotEqual(keys[7], keys[8])
        self.assertEqual(grouped["missing_series_identity"].sum(), 2)

        folds, details = validation_folds(grouped.set_index(grouped.index + 100))
        self.assertEqual(len(folds), 5)
        self.assertEqual(details[0]["purged_train_games"], 1)
        self.assertIn(5, folds[0][1])
        self.assertNotIn(3, folds[0][0])
        for train_idx, valid_idx in folds:
            train, valid = grouped.iloc[train_idx], grouped.iloc[valid_idx]
            self.assertLess(train["date"].max(), valid["date"].min())
            self.assertTrue(set(train["series_group"]).isdisjoint(valid["series_group"]))
            self.assertEqual(len(valid), 2 * valid["date"].nunique())

    def test_identical_predictions_have_zero_paired_brier_difference(self):
        y = np.array([0, 1, 1])
        p = np.array([0.2, 0.9, 0.5])
        result = paired_brier_bootstrap(y, p, p, pd.Series(["A", "A", "B"]))
        self.assertEqual(result["gold_only_minus_five_features"], 0)
        self.assertEqual(result["ci95"], [0, 0])
        self.assertEqual(result["series_groups"], 2)
        self.assertFalse(result["five_features_improved"])

    def test_cli_reports_are_repeatable_and_preserve_patch_labels(self):
        rows = [
            match(2 * i + result, date, result)
            for i, date in enumerate(pd.date_range("2026-02-01", periods=12))
            for result in (0, 1)
        ]
        rows += [
            match(24, TEST_START - pd.Timedelta(minutes=1), 0, "A", "B"),
            match(25, TEST_START, 1, "B", "A"),
            match(26, TEST_START, 0, "C", "D", patch_id="16.10"),
            match(27, TEST_START, 1, "E", "F"),
            match(28, SNAPSHOT_END + pd.Timedelta(seconds=1), 0),
        ]

        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw, output = root / "matches.csv.gz", root / "reports"
            pd.DataFrame(rows).to_csv(raw, index=False)
            output.mkdir()
            legacy = output / "metrics.json"
            legacy.write_text("ilk teslim", encoding="utf-8")
            args = ["evaluate_esports", "--raw", str(raw), "--output-dir", str(output)]
            with patch("sys.argv", args), patch("builtins.print"):
                main()
                metric_path = output / "esports_metrics.json"
                group_path = output / "esports_group_performance.csv"
                first = metric_path.read_bytes(), group_path.read_bytes()
                main()
                self.assertEqual(first, (metric_path.read_bytes(), group_path.read_bytes()))

            self.assertEqual(legacy.read_text(encoding="utf-8"), "ilk teslim")
            legacy_clean, _ = clean(load_raw(raw))
            self.assertNotIn("team_name", legacy_clean)
            self.assertEqual(legacy_clean["patch"].nunique(), 1)

            metrics = json.loads(first[0])
            self.assertEqual(metrics["snapshot"]["raw_sha256"], hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertEqual(metrics["snapshot"]["cleaned_games_outside_window"], 1)
            self.assertEqual(metrics["split"]["purged_train_games"], 1)
            self.assertEqual(metrics["split"]["test"]["games"], 3)
            self.assertEqual(metrics["references"]["constant_blue_probability"], 0.5)
            gold, five = (metrics["models"][name]["test"] for name in ("gold_only", "five_features"))
            self.assertAlmostEqual(metrics["brier_comparison"]["gold_only_minus_five_features"],
                                   gold["brier"] - five["brier"])
            patch_rows = {row["group"]: row for row in csv.DictReader(first[1].decode().splitlines())
                          if row["group_by"] == "patch"}
            self.assertEqual(set(patch_rows), {"16.1", "16.10"})
            self.assertEqual(sum(int(row["n"]) for row in patch_rows.values()), 3)
            self.assertEqual(patch_rows["16.10"]["gold_only_roc_auc"], "")
            self.assertEqual(patch_rows["16.10"]["five_features_roc_auc"], "")
            self.assertEqual(patch_rows["16.10"]["descriptive_only"], "True")
