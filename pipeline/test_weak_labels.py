"""Regression checks for automatic-label uncertainty and artifact isolation."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.gbdt_model import load_model
from pipeline.weak_labels import build_weak_labels


def frames(n=1):
    return pd.DataFrame({
        "half": np.ones(n), "ball_visible": np.ones(n), "ball_x": np.full(n, 0.5),
        "ball_y": np.full(n, 0.5), "ball_speed": np.full(n, 3.0),
        "nearest_home_dist": np.ones(n), "nearest_away_dist": np.full(n, 8.0),
        "st_is_offensive_home": np.zeros(n), "st_is_offensive_away": np.zeros(n),
    })


class WeakLabelTests(unittest.TestCase):
    def test_missing_ball_and_contested_ball_are_unknown(self):
        data = frames(3)
        data.loc[0, "ball_visible"] = 0
        data.loc[1, ["nearest_home_dist", "nearest_away_dist"]] = [1.0, 1.1]
        data.loc[2, "half"] = 0
        self.assertEqual(build_weak_labels(data)["label"].tolist(), [-1, -1, -1])

    def test_opposing_tag_prevents_confident_team_label(self):
        data = frames()
        data.loc[0, "st_is_offensive_away"] = 1
        self.assertEqual(build_weak_labels(data).loc[0, "label"], -1)

    def test_outside_ball_is_dead_even_near_player(self):
        data = frames()
        data.loc[0, "ball_y"] = 1.1
        self.assertEqual(build_weak_labels(data).loc[0, "label"], 2)

    def test_stationary_ball_needs_duration_and_same_half(self):
        data = frames(50)
        data["ball_speed"] = 0
        data["nearest_home_dist"] = 7
        self.assertTrue((build_weak_labels(data)["label"] == 2).all())
        self.assertTrue((build_weak_labels(data.iloc[:49])["label"] == -1).all())
        data.loc[25:, "half"] = 2
        self.assertTrue((build_weak_labels(data)["label"] == -1).all())

    def test_old_supervised_model_is_rejected_before_loading_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "gbdt_meta.json").write_text('{"feature_names": []}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a ground-truth-free model"):
                load_model(folder)


if __name__ == "__main__":
    unittest.main()
