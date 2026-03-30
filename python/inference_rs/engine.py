"""Vision OpenVINO inference engine.

Matches the Rust ``engine.rs`` architecture:
- Read model from XML + BIN
- Build OpenVINO PrePostProcessor pipeline (NHWC input → NCHW model, bilinear resize)
- Compile to target device
- Single-output ``infer()`` → flat f32 list
- Multi-output ``infer_multi()`` → dict of numpy arrays
- Reusable ``InferRequest`` via ``create_request()`` / ``run_request()``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import openvino as ov


class Engine:
    """High-level wrapper around OpenVINO vision inference."""

    def __init__(
        self,
        model_xml: Path | str,
        model_bin: Path | str,
        device: str,
        reference_tensor: np.ndarray,
    ) -> None:
        self._core = ov.Core()
        model = self._core.read_model(str(model_xml), str(model_bin))

        # Discover input/output names.
        self._input_name = model.input(0).get_any_name()
        self._output_names = [
            model.output(i).get_any_name() for i in range(len(model.outputs))
        ]
        if not self._output_names:
            raise RuntimeError("model has no outputs")

        # Build PrePostProcessor pipeline.
        single_output = len(self._output_names) == 1
        ppp = ov.preprocess.PrePostProcessor(model)

        # Input: set element type and layout from reference tensor.
        input_info = ppp.input(self._input_name)
        input_info.tensor().set_element_type(ov.Type.f32).set_layout(ov.Layout("NHWC"))
        input_info.preprocess().resize(ov.preprocess.ResizeAlgorithm.RESIZE_LINEAR)
        input_info.model().set_layout(ov.Layout("NCHW"))

        # For single-output models, coerce output to F32.
        if single_output:
            ppp.output(0).tensor().set_element_type(ov.Type.f32)

        model = ppp.build()
        self._compiled = self._core.compile_model(model, device)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_names(self) -> list[str]:
        return self._output_names

    def infer(self, input_tensor: np.ndarray) -> list[float]:
        """Run inference and return the first output as a flat float list."""
        request = self._compiled.create_infer_request()
        request.set_tensor(self._input_name, ov.Tensor(input_tensor))
        request.infer()
        result = request.get_tensor(self._output_names[0]).data
        return result.flatten().tolist()

    def infer_multi(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """Run inference and return all outputs keyed by name."""
        request = self._compiled.create_infer_request()
        request.set_tensor(self._input_name, ov.Tensor(input_tensor))
        request.infer()
        outputs: dict[str, np.ndarray] = {}
        for name in self._output_names:
            outputs[name] = np.array(request.get_tensor(name).data)
        return outputs

    def create_request(self) -> ov.InferRequest:
        """Create a reusable inference request."""
        return self._compiled.create_infer_request()

    def run_request(self, request: ov.InferRequest, input_tensor: np.ndarray) -> None:
        """Run inference on an existing request (for benchmarking)."""
        request.set_tensor(self._input_name, ov.Tensor(input_tensor))
        request.infer()
