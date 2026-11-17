#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCS 适配层单元测试 — 纯逻辑,不加载 GPU 模型,可离线运行。

用法:
    python test/test_dcs_adapter.py            # 直接运行
    pytest test/test_dcs_adapter.py -v         # pytest 运行

说明:
    测试导入 backend/dcs_adapter.py 中的纯逻辑函数
    (_parse_common / _format_values / _count_elements / _usage / _ok / _err),
    不触发 init_predictor(),因此不需要 GPU 与模型权重。
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent          # rice-mut/test/
ROOT_DIR = TEST_DIR.parent                          # rice-mut/
BACKEND_DIR = ROOT_DIR / "backend"                  # rice-mut/backend/

# 确保能 import dcs_adapter(其内部会往 sys.path 加 backend/)
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import dcs_adapter  # noqa: E402
from dcs_adapter import (  # noqa: E402
    _check_api_key,
    _count_elements,
    _err,
    _format_values,
    _ok,
    _parse_common,
    _unauthorized,
    _usage,
)


class TestParseCommon(unittest.TestCase):
    """公共参数解析:坐标 1-based 输入,返回 0-based 约定。"""

    def test_minimal(self):
        genome, chrom, start_1, end_0, bios, fmt, max_points = _parse_common(
            {"genome": "osa1_r7", "chromosome": "chr01", "start": 1000}
        )
        self.assertEqual(genome, "osa1_r7")
        self.assertEqual(chrom, "chr01")
        self.assertEqual(start_1, 1000)   # 1-based
        self.assertIsNone(end_0)          # end 省略 → None(交给 run_prediction_core 处理)

    def test_end_kept_as_is(self):
        # end 传 1-based inclusive;适配层不做减 1(end_0 直接透传,窗口对齐由内部处理)
        _, _, _, end_0, _, _, _ = _parse_common(
            {"genome": "osa1_r7", "start": 20716774, "end": 20749541}
        )
        self.assertEqual(end_0, 20749541)

    def test_biosample_names_list_and_csv(self):
        _, _, _, _, bios, _, _ = _parse_common(
            {"genome": "osa1_r7", "start": 1, "biosample_names": ["Leaf"]}
        )
        self.assertEqual(bios, ["Leaf"])
        _, _, _, _, bios2, _, _ = _parse_common(
            {"genome": "osa1_r7", "start": 1, "biosample_names": " Leaf , Root "}
        )
        self.assertEqual(bios2, ["Leaf", "Root"])

    def test_biosample_names_absent(self):
        _, _, _, _, bios, _, _ = _parse_common({"genome": "osa1_r7", "start": 1})
        self.assertIsNone(bios)

    def test_output_format_default_and_valid(self):
        for fmt in ("full", "mean", "downsample"):
            _, _, _, _, _, f, _ = _parse_common(
                {"genome": "osa1_r7", "start": 1, "output_format": fmt}
            )
            self.assertEqual(f, fmt)
        _, _, _, _, _, f, _ = _parse_common({"genome": "osa1_r7", "start": 1})
        self.assertEqual(f, "full")

    def test_output_format_invalid(self):
        with self.assertRaises(ValueError):
            _parse_common({"genome": "osa1_r7", "start": 1, "output_format": "bogus"})

    def test_missing_start(self):
        with self.assertRaises(ValueError):
            _parse_common({"genome": "osa1_r7"})

    def test_max_points_default(self):
        _, _, _, _, _, _, mp = _parse_common({"genome": "osa1_r7", "start": 1})
        self.assertEqual(mp, 1024)
        _, _, _, _, _, _, mp2 = _parse_common(
            {"genome": "osa1_r7", "start": 1, "max_points": 512}
        )
        self.assertEqual(mp2, 512)


class TestFormatValues(unittest.TestCase):
    """output_format: full / mean / downsample 输出结构。"""

    def _values(self):
        return {"RNA-seq": {"Leaf": np.array([0.1234567, 0.5, 1.0], dtype=np.float64)}}

    def test_full_rounds_to_6(self):
        out = _format_values(self._values(), "full", 1024)
        self.assertEqual(out["RNA-seq"]["Leaf"], [0.123457, 0.5, 1.0])

    def test_mean_scalar(self):
        out = _format_values(self._values(), "mean", 1024)
        self.assertAlmostEqual(out["RNA-seq"]["Leaf"], (0.1234567 + 0.5 + 1.0) / 3, places=6)
        self.assertIsInstance(out["RNA-seq"]["Leaf"], float)

    def test_downsample_shorter_than_max_kept(self):
        out = _format_values(self._values(), "downsample", 1024)
        self.assertEqual(len(out["RNA-seq"]["Leaf"]), 3)

    def test_downsample_reduces(self):
        big = {"A": {"B": np.arange(10000, dtype=np.float64)}}
        out = _format_values(big, "downsample", 100)
        self.assertEqual(len(out["A"]["B"]), 100)

    def test_downsample_covers_ends(self):
        arr = np.arange(1000, dtype=np.float64)
        out = _format_values({"A": {"B": arr}}, "downsample", 50)
        vals = out["A"]["B"]
        self.assertEqual(vals[0], 0.0)
        self.assertEqual(vals[-1], 999.0)


class TestCountElements(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_count_elements({}), 0)

    def test_multiple_tracks(self):
        v = {
            "RNA-seq": {"Leaf": np.zeros(10), "Root": np.zeros(5)},
            "ATAC": {"Leaf": np.zeros(3)},
        }
        self.assertEqual(_count_elements(v), 18)


class TestUsageAndResponses(unittest.TestCase):
    def test_usage_default_multiplier(self):
        u = _usage(32768, 32768)
        self.assertEqual(u, {"prompt_tokens": 32768, "completion_tokens": 32768})

    def test_usage_multiplier_env(self):
        # _usage 使用模块级常量(import 时读取 env),这里 monkeypatch 常量本身
        orig_p, orig_c = dcs_adapter.PROMPT_TOKEN_MULTIPLIER, dcs_adapter.COMPLETION_TOKEN_MULTIPLIER
        try:
            dcs_adapter.PROMPT_TOKEN_MULTIPLIER = 2
            dcs_adapter.COMPLETION_TOKEN_MULTIPLIER = 0.5
            u = _usage(1000, 1000)
            self.assertEqual(u, {"prompt_tokens": 2000, "completion_tokens": 500})
        finally:
            dcs_adapter.PROMPT_TOKEN_MULTIPLIER = orig_p
            dcs_adapter.COMPLETION_TOKEN_MULTIPLIER = orig_c

    def test_ok_shape(self):
        resp = _ok({"prompt_tokens": 1, "completion_tokens": 2}, "ok", {"k": "v"})
        self.assertEqual(resp["status"], 200)
        self.assertEqual(resp["message"], "ok")
        self.assertEqual(resp["result"], {"k": "v"})

    def test_err_shape(self):
        resp = _err("bad", 400)
        self.assertEqual(resp["status"], 400)
        self.assertIsNone(resp["result"])
        self.assertEqual(resp["usage"], {"prompt_tokens": 0, "completion_tokens": 0})


class TestSnvBilling(unittest.TestCase):
    """SNV 计费:completion_tokens = ref + mut 元素总数。"""

    def test_snv_style_count(self):
        ref = {"RNA-seq": {"Leaf": np.zeros(32768)}}
        mut = {"RNA-seq": {"Leaf": np.zeros(32768)}}
        total = _count_elements(ref) + _count_elements(mut)
        self.assertEqual(total, 65536)


class TestApiKey(unittest.TestCase):
    """API Key 鉴权:_check_api_key 逻辑(离线,不依赖路由)。

    注意:DCS_API_KEY 是 import 时从环境变量读取的模块级常量,
    这里直接 monkeypatch 模块属性,与 TestUsageAndResponses 相同模式。
    """

    def setUp(self):
        self._orig = dcs_adapter.DCS_API_KEY

    def tearDown(self):
        dcs_adapter.DCS_API_KEY = self._orig

    def test_no_key_configured_passes(self):
        dcs_adapter.DCS_API_KEY = ""
        # 未配置 key → 任何请求头都放行
        _check_api_key(None, None)
        _check_api_key("Bearer whatever", None)
        _check_api_key(None, "whatever")

    def test_missing_header_raises(self):
        dcs_adapter.DCS_API_KEY = "secret-key-123"
        with self.assertRaises(ValueError):
            _check_api_key(None, None)

    def test_wrong_bearer_raises(self):
        dcs_adapter.DCS_API_KEY = "secret-key-123"
        with self.assertRaises(ValueError):
            _check_api_key("Bearer wrong-key", None)
        with self.assertRaises(ValueError):
            _check_api_key("Basic dXNlcjpwYXNz", None)  # 非 Bearer 前缀

    def test_correct_bearer_and_x_api_key_pass(self):
        dcs_adapter.DCS_API_KEY = "secret-key-123"
        _check_api_key("Bearer secret-key-123", None)
        _check_api_key(None, "secret-key-123")
        # 大小写不敏感前缀:Bearer 前后空白容忍
        _check_api_key("  bearer secret-key-123  ", None)

    def test_unauthorized_shape(self):
        resp = _unauthorized()
        self.assertEqual(resp["status"], 401)
        self.assertIsNone(resp["result"])
        self.assertEqual(resp["usage"], {"prompt_tokens": 0, "completion_tokens": 0})


if __name__ == "__main__":
    unittest.main(verbosity=2)
