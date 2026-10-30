"""CPU regression checks for V4.1 language-only source loading."""
import torch
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcore_bridge.bridge import GPTBridge
from mcore_bridge.model.mm_gpts.deepseek_v41 import DeepseekV41Bridge


class SourceLoadingTest(unittest.TestCase):

    def test_block32_fp8_source_copies_to_bf16(self):
        bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
        bridge.config = SimpleNamespace(hf_config=SimpleNamespace(quantization_config={'weight_block_size': [32, 32]}))
        source = torch.ones(64, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[1., 2.], [3., 4.]])
        param = torch.nn.Parameter(torch.zeros(64, 64, dtype=torch.bfloat16))
        expected = scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
        with patch.object(bridge, '_is_fp8_param', return_value=False):
            bridge._set_param(param, source, scales)
        torch.testing.assert_close(param.detach().float(), expected)
        with patch.object(bridge, '_is_fp8_param', return_value=True):
            with self.assertRaisesRegex(ValueError, 'BF16 training parameters'):
                bridge._set_param(param, source, scales)

    def test_engram_block32_and_row_scales(self):
        weight = torch.ones(64, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[1., 2.], [3., 4.]])
        result = DeepseekV41Bridge._dequantize_engram_rows(weight, scales)
        torch.testing.assert_close(result, scales.repeat_interleave(32, 0).repeat_interleave(32, 1))
        row_scale = torch.tensor([[2., 3.]]).expand(64, 2)
        result = DeepseekV41Bridge._dequantize_engram_rows(weight, row_scale)
        torch.testing.assert_close(result, row_scale.repeat_interleave(32, 1))
        self.assertIs(DeepseekV41Bridge._dequantize_engram_rows(weight, None), weight)
        with self.assertRaises(ValueError):
            DeepseekV41Bridge._dequantize_engram_rows(weight, torch.ones(3, 3))

    def test_language_only_preprocessing_skips_vision_markers(self):
        bridge = DeepseekV41Bridge.__new__(DeepseekV41Bridge)
        bridge.config = SimpleNamespace(language_model_only=True)
        with patch.object(GPTBridge, '_convert_pre_process', return_value={'embedding': 1}) as native:
            result = bridge._convert_pre_process(object(), {}, '', True)
        self.assertEqual(result, {'embedding': 1})
        native.assert_called_once()


if __name__ == '__main__':
    unittest.main()
