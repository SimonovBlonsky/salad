"""Model components for SALAD ablations."""

from .dinov2_intermediate import DINOv2IntermediateEncoder
from .vpr_helper import (
    AGGREGATOR_STATE_PREFIXES,
    build_aggregator,
    build_dino_encoder,
    build_vpr_model,
    extract_prefixed_state_dict,
    load_aggregator_weights_from_salad_ckpt,
)
from .vpr_model import VPRModel
