from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig


CHAR_TO_TOKEN: Dict[str, int] = {
    **{str(i): i for i in range(10)},
    "+": 10,
    "-": 11,
    "=": 12,
    "*": 13,
    "\\": 14,
}
TOKEN_TO_CHAR: Dict[int, str] = {token_id: ch for ch, token_id in CHAR_TO_TOKEN.items()}
DIGIT_TOKEN_IDS = tuple(range(10))


def resolve_model_dir(model_path_or_dir: Union[str, Path]) -> Path:
    path = Path(model_path_or_dir).expanduser().resolve()
    if path.is_file():
        if path.name != "model.pth":
            raise ValueError(f"Unsupported model file path: {path}")
        path = path.parent
    model_path = path / "model.pth"
    training_path = path / "training_loss.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights file: {model_path}")
    if not training_path.exists():
        raise FileNotFoundError(f"Missing training config file: {training_path}")
    return path


def load_training_config(model_path_or_dir: Union[str, Path]) -> Dict[str, int]:
    model_dir = resolve_model_dir(model_path_or_dir)
    training_path = model_dir / "training_loss.json"
    data = json.loads(training_path.read_text())
    if "Config" not in data:
        raise KeyError(f"Missing Config field in {training_path}")
    return data["Config"]


def _infer_n_ctx(config_data: Dict[str, int]) -> int:
    n_digits = int(config_data["n_digits"])
    return 3 * n_digits + 4


def build_hooked_transformer_config(
    model_path_or_dir: Union[str, Path],
    device: Union[str, torch.device, None] = None,
) -> Tuple[HookedTransformerConfig, Dict[str, int]]:
    config_data = load_training_config(model_path_or_dir)
    device_str = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    d_model = int(config_data.get("d_model", config_data["n_heads"] * config_data["d_head"]))
    d_mlp = int(config_data.get("d_mlp", d_model * config_data.get("d_mlp_multiplier", 4)))

    ht_cfg = HookedTransformerConfig(
        n_layers=int(config_data["n_layers"]),
        n_heads=int(config_data["n_heads"]),
        d_model=d_model,
        d_head=int(config_data["d_head"]),
        d_mlp=d_mlp,
        act_fn=str(config_data["act_fn"]),
        normalization_type="LN",
        d_vocab=int(config_data["d_vocab"]),
        d_vocab_out=int(config_data["d_vocab"]),
        n_ctx=int(config_data.get("n_ctx", _infer_n_ctx(config_data))),
        init_weights=True,
        device=device_str,
        seed=int(config_data.get("training_seed", 0)),
    )
    return ht_cfg, config_data


def _safe_torch_load(path: Path, map_location: Union[str, torch.device] = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_quanta_model(
    model_path_or_dir: Union[str, Path],
    device: Union[str, torch.device, None] = None,
):
    model_dir = resolve_model_dir(model_path_or_dir)
    ht_cfg, config_data = build_hooked_transformer_config(model_dir, device=device)
    model = HookedTransformer(ht_cfg)
    state_dict = _safe_torch_load(model_dir / "model.pth", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(target_device)
    model.cfg.default_prepend_bos = False
    model.eval()
    # Expose device for compatibility with flow_utils / umap_plot_script.
    model.device = target_device
    return model, {
        "model_dir": str(model_dir),
        "config": config_data,
        "n_ctx": ht_cfg.n_ctx,
        "n_layers": ht_cfg.n_layers,
        "n_digits": int(config_data["n_digits"]),
    }


def encode_text(text: str) -> torch.Tensor:
    return torch.tensor([CHAR_TO_TOKEN[ch] for ch in text], dtype=torch.long)


def decode_token_ids(token_ids: Iterable[int]) -> str:
    chars = []
    for token_id in token_ids:
        chars.append(TOKEN_TO_CHAR.get(int(token_id), "?"))
    return "".join(chars)


def encode_prompt(a: int, b: int, n_digits: int) -> Tuple[str, torch.Tensor]:
    prompt = f"{a:0{n_digits}d}+{b:0{n_digits}d}="
    return prompt, encode_text(prompt)


def answer_to_string(value: int, n_digits: int) -> str:
    sign = "+" if value >= 0 else "-"
    digits = f"{abs(value):0{n_digits + 1}d}"
    return f"{sign}{digits}"


def get_digit_anchors(model, digits: Iterable[int] = DIGIT_TOKEN_IDS) -> torch.Tensor:
    digit_ids = list(digits)
    return model.unembed.W_U[:, digit_ids].detach().float().T.contiguous()
