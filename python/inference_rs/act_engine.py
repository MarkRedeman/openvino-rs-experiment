"""ACT (Action Chunking Transformer) inference engine.

Matches the Rust ``act.rs`` architecture:
- 3 inputs: ``state`` [1, state_dim], ``images.gripper`` [1,3,H,W], ``images.overview`` [1,3,H,W]
- No OpenVINO PrePostProcessor — raw tensors fed directly
- Output: ``action`` tensor [1, chunk_size, action_dim]
- Metadata discovery from model input/output shapes
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

from .preprocessing import image_to_nchw


@dataclass
class ActMetadata:
    state_dim: int
    image_height: int
    image_width: int
    camera_names: list[str]
    action_dim: int
    chunk_size: int


@dataclass
class ActInputs:
    state: list[float]
    gripper_image: Image.Image
    overview_image: Image.Image


@dataclass
class ActOutput:
    chunk_size: int
    action_dim: int
    actions: list[list[float]]


class ActEngine:
    """High-level wrapper around OpenVINO ACT model inference."""

    def __init__(
        self,
        model_xml: Path | str,
        model_bin: Path | str,
        device: str,
    ) -> None:
        self._core = ov.Core()
        model = self._core.read_model(str(model_xml), str(model_bin))
        self._metadata = _discover_act_metadata(model)
        self._compiled = self._core.compile_model(model, device)

    @property
    def metadata(self) -> ActMetadata:
        return self._metadata

    def create_request(self) -> ov.InferRequest:
        return self._compiled.create_infer_request()

    def run_request(
        self,
        request: ov.InferRequest,
        state: np.ndarray,
        gripper: np.ndarray,
        overview: np.ndarray,
    ) -> None:
        """Set the 3 ACT input tensors and run inference."""
        request.set_tensor("state", ov.Tensor(state))
        request.set_tensor("images.gripper", ov.Tensor(gripper))
        request.set_tensor("images.overview", ov.Tensor(overview))
        request.infer()


def _discover_act_metadata(model: ov.Model) -> ActMetadata:
    """Discover ACT metadata from model input/output shapes."""
    state_dim: int | None = None
    gripper_hw: tuple[int, int] | None = None
    overview_hw: tuple[int, int] | None = None
    action_shape: tuple[int, int] | None = None

    for inp in model.inputs:
        name = inp.get_any_name()
        shape = inp.shape
        if name == "state":
            if len(shape) != 2 or shape[0] != 1:
                raise RuntimeError(
                    f"unexpected ACT input shape for 'state': {shape} (expected [1, state_dim])"
                )
            state_dim = shape[1]
        elif name == "images.gripper":
            if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
                raise RuntimeError(
                    f"unexpected ACT input shape for 'images.gripper': {shape} (expected [1, 3, H, W])"
                )
            gripper_hw = (shape[2], shape[3])
        elif name == "images.overview":
            if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
                raise RuntimeError(
                    f"unexpected ACT input shape for 'images.overview': {shape} (expected [1, 3, H, W])"
                )
            overview_hw = (shape[2], shape[3])

    for out in model.outputs:
        name = out.get_any_name()
        if name == "action":
            shape = out.shape
            if len(shape) != 3 or shape[0] != 1:
                raise RuntimeError(
                    f"unexpected ACT output shape for 'action': {shape} (expected [1, chunk_size, action_dim])"
                )
            action_shape = (shape[1], shape[2])

    if state_dim is None:
        raise RuntimeError("ACT model is missing input 'state'")
    if gripper_hw is None:
        raise RuntimeError("ACT model is missing input 'images.gripper'")
    if overview_hw is None:
        raise RuntimeError("ACT model is missing input 'images.overview'")
    if gripper_hw != overview_hw:
        raise RuntimeError(
            f"camera shapes differ (gripper={gripper_hw[1]}x{gripper_hw[0]}, "
            f"overview={overview_hw[1]}x{overview_hw[0]}), unsupported"
        )
    if action_shape is None:
        raise RuntimeError("ACT model is missing output 'action'")

    return ActMetadata(
        state_dim=state_dim,
        image_height=gripper_hw[0],
        image_width=gripper_hw[1],
        camera_names=["gripper", "overview"],
        action_dim=action_shape[1],
        chunk_size=action_shape[0],
    )


def prepare_act_tensors(
    meta: ActMetadata,
    inputs: ActInputs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare the 3 ACT input tensors from high-level inputs.

    Returns ``(state_tensor, gripper_tensor, overview_tensor)`` as numpy arrays.
    """
    # State: [1, state_dim] f32
    state = np.array(inputs.state, dtype=np.float32).reshape(1, -1)
    if state.shape[1] != meta.state_dim:
        raise RuntimeError(
            f"state length {state.shape[1]} does not match model state dim {meta.state_dim}"
        )

    # Images: [1, 3, H, W] f32 NCHW
    gripper = image_to_nchw(inputs.gripper_image, meta.image_width, meta.image_height)
    overview = image_to_nchw(inputs.overview_image, meta.image_width, meta.image_height)

    return state, gripper, overview


def read_act_output(request: ov.InferRequest, meta: ActMetadata) -> ActOutput:
    """Read and reshape the ACT output tensor from an inference request."""
    output_data = np.array(request.get_tensor("action").data, dtype=np.float32)
    expected = meta.chunk_size * meta.action_dim
    flat = output_data.flatten()
    if flat.size != expected:
        raise RuntimeError(
            f"unexpected ACT output length: got {flat.size}, expected {expected} "
            f"({meta.chunk_size}x{meta.action_dim})"
        )

    actions = flat.reshape(meta.chunk_size, meta.action_dim).tolist()
    return ActOutput(
        chunk_size=meta.chunk_size,
        action_dim=meta.action_dim,
        actions=actions,
    )


def parse_state_from_episode(data_jsonl: Path | str) -> list[float]:
    """Parse the state vector from the first line of an episode ``data.jsonl``."""
    path = Path(data_jsonl)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                return row["state"]
    raise RuntimeError(f"episode data file is empty: {path}")


def load_sample_images_from_episode(
    episode_dir: Path | str,
    width: int,
    height: int,
) -> dict[str, Image.Image]:
    """Load sample camera images from an episode directory.

    Tries ``cam_gripper.jpg`` / ``cam_gripper.png`` first. Falls back to
    generating uniform-colour images from ``stats.json`` camera means.
    """
    episode_dir = Path(episode_dir)

    gripper_img: Image.Image | None = None
    overview_img: Image.Image | None = None

    for ext in (".jpg", ".png"):
        gripper_path = episode_dir / f"cam_gripper{ext}"
        if gripper_path.exists() and gripper_img is None:
            gripper_img = Image.open(gripper_path).convert("RGB")
        overview_path = episode_dir / f"cam_overview{ext}"
        if overview_path.exists() and overview_img is None:
            overview_img = Image.open(overview_path).convert("RGB")

    if gripper_img is None or overview_img is None:
        # Fall back to stats.json mean colour images
        stats_path = episode_dir / "stats.json"
        with stats_path.open() as f:
            stats = json.load(f)

        if gripper_img is None:
            mean_rgb = stats["images"]["gripper"]["mean"]
            gripper_img = _make_uniform_rgb(width, height, mean_rgb)
        if overview_img is None:
            mean_rgb = stats["images"]["overview"]["mean"]
            overview_img = _make_uniform_rgb(width, height, mean_rgb)

    return {"gripper": gripper_img, "overview": overview_img}


def _make_uniform_rgb(
    width: int, height: int, mean_rgb_0_1: list[float]
) -> Image.Image:
    """Create a uniform-colour image from mean RGB values in [0, 1]."""
    if len(mean_rgb_0_1) != 3:
        raise RuntimeError(
            f"camera mean length {len(mean_rgb_0_1)} is invalid, expected 3"
        )
    r = int(max(0.0, min(1.0, mean_rgb_0_1[0])) * 255 + 0.5)
    g = int(max(0.0, min(1.0, mean_rgb_0_1[1])) * 255 + 0.5)
    b = int(max(0.0, min(1.0, mean_rgb_0_1[2])) * 255 + 0.5)
    return Image.new("RGB", (width, height), (r, g, b))
