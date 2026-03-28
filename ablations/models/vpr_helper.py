from __future__ import annotations

import warnings
from pathlib import Path
from collections.abc import Iterable, Mapping

import torch

from ablations.models.dinov2_intermediate import DINOv2IntermediateEncoder
from ablations.models.vpr_model import VPRModel
from models.aggregators import CosPlace, ConvAP, GeMPool, MixVPR, SALAD


AGGREGATOR_STATE_PREFIXES = (
    "aggregator.",
    "model.aggregator.",
    "module.aggregator.",
)


def build_aggregator(agg_arch, agg_config=None):
    config = {} if agg_config is None else dict(agg_config)
    name = str(agg_arch).lower()

    if name == "cosplace":
        return CosPlace(**config)
    if name in {"gem", "gempool"}:
        config.setdefault("p", 3)
        return GeMPool(**config)
    if name == "convap":
        return ConvAP(**config)
    if name == "mixvpr":
        return MixVPR(**config)
    if name == "salad":
        return SALAD(**config)

    raise ValueError(f"Unsupported aggregator: {agg_arch}")


def extract_prefixed_state_dict(
    state_dict: Mapping[str, torch.Tensor], prefixes: Iterable[str]
):
    extracted = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                extracted[key[len(prefix) :]] = value
                break
    return extracted


def _unwrap_checkpoint_mapping(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected checkpoint to contain a mapping")
    if isinstance(checkpoint.get("state_dict"), Mapping):
        return checkpoint["state_dict"]
    if isinstance(checkpoint.get("model"), Mapping):
        return checkpoint["model"]
    return checkpoint


def load_aggregator_weights_from_salad_ckpt(aggregator, ckpt_path, strict=True):
    checkpoint = torch.load(Path(ckpt_path), map_location="cpu")
    state_dict = _unwrap_checkpoint_mapping(checkpoint)
    aggregator_state_dict = extract_prefixed_state_dict(
        state_dict, AGGREGATOR_STATE_PREFIXES
    )
    if not aggregator_state_dict:
        raise ValueError("No aggregator-prefixed keys found in SALAD checkpoint")
    incompatible_keys = aggregator.load_state_dict(aggregator_state_dict, strict=strict)
    if not strict and incompatible_keys.unexpected_keys:
        unexpected = ", ".join(sorted(incompatible_keys.unexpected_keys))
        raise ValueError(
            "incompatible aggregator checkpoint for target aggregator: "
            f"unexpected keys {unexpected}"
        )
    if not strict and incompatible_keys.missing_keys:
        missing = ", ".join(sorted(incompatible_keys.missing_keys))
        warnings.warn(
            "partial/incomplete aggregator checkpoint load; missing keys: "
            f"{missing}",
            UserWarning,
            stacklevel=2,
        )
    return aggregator


def build_dino_encoder(**kwargs):
    return DINOv2IntermediateEncoder(**kwargs)


def build_vpr_model(
    *,
    agg_arch,
    agg_config=None,
    aggregator_ckpt_path=None,
    strict=True,
    **encoder_kwargs,
):
    aggregator = build_aggregator(agg_arch, agg_config=agg_config)
    if aggregator_ckpt_path is not None:
        load_aggregator_weights_from_salad_ckpt(
            aggregator,
            aggregator_ckpt_path,
            strict=strict,
        )
    encoder = build_dino_encoder(**encoder_kwargs)
    model = VPRModel(encoder=encoder, aggregator=aggregator, agg_arch=agg_arch)
    model.eval()
    return model
