"""Regression tests for batched matched interventions and paired reporting."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from where_to_write.probe import ARMS, ArmResult, QwenProbe, top_token_is_color
from where_to_write.benchmark import run


class MockProbe:
    model_name = 'mock'
    layer = 16
    direction_kind = 'mock'

    def evaluate(self, name, color, template, gain):
        return {
            arm: ArmResult(arm, ' Blue', arm == 'answer',
                           color if arm in ('subject', 'answer') else 'red',
                           arm in ('subject', 'answer'),
                           {}, 1.0 if arm in ('subject', 'answer') else -1.0,
                           arm == 'answer') for arm in ARMS
        }


class BatchTests(unittest.TestCase):
    def test_semantic_color(self):
        self.assertTrue(top_token_is_color(' Blue', 'blue'))
        self.assertFalse(top_token_is_color(' blueberry', 'blue'))

    def test_paired_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            report = run(MockProbe(), trials=24, out=str(Path(td) / 'report.json'))
            self.assertEqual(report['paired_site_comparison']['subject_only_correct'], 24)
            self.assertEqual(report['paired_site_comparison']['color_token_only_correct'], 0)
            self.assertEqual(report['paired_site_comparison']['subject_minus_color_token_accuracy'], 1)
            self.assertEqual(report['conditions']['answer']['semantic_top_color_accuracy'], 1)

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None, 'PyTorch optional in CI')
    def test_batched_equals_single_forwards(self):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(17)

        class TinyFrozen(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(20, 8)
                self.block = torch.nn.Linear(8, 8)
                self.unembed = torch.nn.Linear(8, 20)

            def forward(self, input_ids, use_cache=False):
                x = self.block(self.embed(input_ids))
                return SimpleNamespace(logits=self.unembed(x))

        probe = QwenProbe.__new__(QwenProbe)
        probe.torch = torch
        probe.model = TinyFrozen().eval()
        probe.layers = (probe.model.block,)
        probe.layer = 0
        probe.directions = F.normalize(torch.randn(4, 8), dim=1)
        probe.random_direction = F.normalize(torch.randn(8), dim=0)
        encoded = {'input_ids': torch.tensor([[1, 3, 5, 7, 9]])}
        for gain in (0.5, 2.0, 8.0):
            batch = probe._logits_many(encoded, ARMS, 'blue', 2, 1, gain)
            for arm in ARMS:
                single = probe._logits(encoded, arm, 'blue', 2, 1, gain)
                self.assertTrue(torch.allclose(batch[arm], single, atol=1e-5, rtol=1e-5))
