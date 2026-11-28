import importlib.util
import unittest

from src.artifacts import image_weighting, run_directory, training_config
from src.config import CLASSES


class WeightingTests(unittest.TestCase):
    def test_preserves_seed42_control_identity_and_separates_treatment(self):
        args = ["--model", "multimodal", "--recipe", "v2", "--seed", "42", "--workers", "6", "--tag", "track2"]
        control, _ = training_config(args)
        weighted, _ = training_config(args + ["--image-class-weighting", "capped-sqrt"])
        self.assertEqual(run_directory(control).name, "multimodal_seed42-34631196455259ac")
        self.assertEqual(weighted, {**control, "image_class_weighting": "capped-sqrt"})
        self.assertNotEqual(run_directory(control), run_directory(weighted))

    def test_training_counts_and_clipping(self):
        support = [498, 2991, 6565, 1860, 1694]
        labels = [label for label, count in zip(CLASSES, support) for _ in range(count)]
        result = image_weighting(labels)
        self.assertEqual(result["counts"], dict(zip(CLASSES, support)))
        self.assertEqual(result["weights"][CLASSES[0]], 1.5)
        self.assertEqual(result["weights"][CLASSES[2]], 0.75)
        self.assertAlmostEqual(result["weights"][CLASSES[1]], (2721.6 / 2991) ** .5)
        with self.assertRaisesRegex(ValueError, "positive train support"):
            image_weighting(labels[498:])

    @unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("torch", "pandas", "transformers", "torchvision",
                                                                   "sklearn")), "training dependencies unavailable")
    def test_weighting_changes_only_smoothed_image_training_loss(self):
        import torch
        import torch.nn.functional as F
        from src.train import compute_loss

        out = {"text": torch.zeros(2, 5), "image": torch.tensor([[0., 3., 0., 0., 0.], [0., 0., 1., 0., 0.]]),
               "agree": torch.tensor([1., -1.])}
        targets = {"text_label": torch.tensor([0, 0]), "image_label": torch.tensor([0, 2]),
                   "agree": torch.tensor([1., 0.])}
        weights = torch.tensor([1.5, .9, .75, 1.2, 1.3])
        unweighted = F.cross_entropy(out["image"], targets["image_label"], label_smoothing=.1)
        weighted = F.cross_entropy(out["image"], targets["image_label"], weight=weights, label_smoothing=.1)
        difference = compute_loss(out, targets, .1, weights) - compute_loss(out, targets, .1)
        self.assertTrue(torch.allclose(difference, weighted - unweighted, atol=1e-6))
        self.assertGreater(float(difference), 0)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn unavailable")
class DiagnosticsTests(unittest.TestCase):
    def test_alignment_uses_ids_and_rejects_target_mismatch(self):
        import numpy as np
        from src.diagnose import aligned

        a = {k: np.array(v) for k, v in {"tweet_id": ["t1", "t2"], "image_id": ["i1", "i2"],
                                        "y_text_label": [0, 1], "y_image_label": [0, 2], "y_agree": [1, 0]}.items()}
        b = {key: value[::-1].copy() for key, value in a.items()}
        self.assertEqual(aligned(a, b), [1, 0])
        b["y_agree"][0] = 1
        with self.assertRaisesRegex(ValueError, "Targets differ"):
            aligned(a, b)
        b["image_id"][:] = "i1"
        with self.assertRaisesRegex(ValueError, "IDs are duplicated or differ"):
            aligned(a, b)

    def test_fixed_threshold_and_support_counts(self):
        import numpy as np
        from src.diagnose import summary

        a = {"text": np.eye(5)[[0, 1, 0]], "image": np.eye(5)[[0, 1, 1]], "agree": np.array([0., -1., -1.]),
             "y_text_label": np.array([0, 1, 0]), "y_image_label": np.array([0, 1, 1]), "y_agree": np.array([1, 1, 0])}
        result = summary(a, np.ones(3, bool))
        self.assertEqual([result["warning"][key] for key in ("tn", "fp", "fn", "tp")], [1, 1, 0, 1])
        self.assertEqual(result["classification"]["all"]["image"]["classes"][CLASSES[1]]["support"], 2)


if __name__ == "__main__":
    unittest.main()
