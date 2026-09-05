# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestExp2(PytorchLayerTest):
    def _prepare_input(
        self, input_shape=(2, 3), dtype="float32", out=False, edge_cases=False
    ):
        if edge_cases:
            inp = np.array(
                [
                    -float("inf"),
                    -150.0,
                    -10.0,
                    -1.0,
                    0.0,
                    1.0,
                    10.0,
                    128.0,
                    float("inf"),
                    float("nan"),
                ],
                dtype=np.float32,
            )
            return (inp,)
        if dtype == "bool":
            inp = self.random.randint(0, 2, size=input_shape).astype(bool)
        elif "int" in dtype:
            inp = self.random.randint(-10, 10, size=input_shape).astype(dtype)
        else:
            inp = self.random.uniform(-10.0, 10.0, size=input_shape).astype(dtype)
        if not out:
            return (inp,)
        out_tensor = np.zeros(
            input_shape, dtype=dtype if "float" in dtype else "float32"
        )
        return (inp, out_tensor)

    def create_model(self, variant="default"):
        class aten_exp2(torch.nn.Module):
            def forward(self, x):
                return torch.exp2(x)

        class aten_exp2_inplace(torch.nn.Module):
            def forward(self, x):
                return x.exp2_()

        class aten_exp2_out(torch.nn.Module):
            def forward(self, x, out):
                return torch.exp2(x, out=out), out

        models = {
            "default": aten_exp2,
            "inplace": aten_exp2_inplace,
            "out": aten_exp2_out,
        }
        suffix = "_" if variant == "inplace" else ""
        op_name = f"aten::exp2{suffix}"
        return models[variant](), op_name

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("input_shape", [(), (5,), (2, 3), (1, 3, 8, 8)])
    @pytest.mark.parametrize(
        "dtype",
        [
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uint8",
            "bool",
        ],
    )
    def test_exp2(self, input_shape, dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model("default"),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_shape": input_shape, "dtype": dtype},
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [(2, 3), (1, 3, 8, 8)])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize(
        "variant", [skip_if_export("inplace"), skip_if_export("out")]
    )
    def test_exp2_variants(
        self, input_shape, dtype, variant, ie_device, precision, ir_version
    ):
        is_out = variant == "out"
        self._test(
            *self.create_model(variant),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input_shape": input_shape,
                "dtype": dtype,
                "out": is_out,
            },
            rtol=1e-4,
            atol=1e-4,
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_exp2_edge_cases(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model("default"),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"edge_cases": True},
            rtol=1e-4,
            atol=1e-4,
        )
