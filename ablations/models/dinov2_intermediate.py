from __future__ import annotations

import torch
import torch.nn as nn


DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2IntermediateEncoder(nn.Module):
    def __init__(
        self,
        model_name="dinov2_vitb14",
        output_layer=-1,
        freeze_backbone=False,
        num_trainable_blocks=4,
        norm_layer=True,
        return_token=True,
        patch_size=14,
        hub_loader=None,
    ):
        super().__init__()

        if model_name not in DINOV2_ARCHS:
            raise ValueError(f"Unknown model name {model_name}")

        self.num_channels = DINOV2_ARCHS[model_name]
        self.patch_size = patch_size
        self.output_layer = output_layer
        self.freeze_backbone = freeze_backbone
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        # Compatibility with the parent encoder API: this ablation always
        # returns the DA3-style feature dict, including `global_token`.
        self.return_token = return_token

        loader = hub_loader if hub_loader is not None else torch.hub.load
        self.model = loader("facebookresearch/dinov2", model_name)

        if not hasattr(self.model, "blocks"):
            raise ValueError("Loaded DINOv2 model must expose blocks")
        if not hasattr(self.model, "norm"):
            raise ValueError("Loaded DINOv2 model must expose norm")
        if not hasattr(self.model, "prepare_tokens_with_masks"):
            raise ValueError("Loaded DINOv2 model must expose prepare_tokens_with_masks")

        self.num_blocks = len(self.model.blocks)
        self._validate_output_layer()
        self._configure_trainable_parameters()

    def _validate_output_layer(self):
        if self.output_layer == -1:
            return
        if not isinstance(self.output_layer, int):
            raise ValueError("output_layer must be an int")
        if self.output_layer < 0 or self.output_layer >= self.num_blocks:
            raise ValueError(
                f"Invalid output_layer {self.output_layer}; expected -1 or a value in [0, {self.num_blocks - 1}]"
            )

    def _is_fully_frozen(self):
        return self.freeze_backbone or self.num_trainable_blocks <= 0

    def _selected_block_index(self):
        return self.num_blocks - 1 if self.output_layer == -1 else self.output_layer

    def _execution_bounds(self):
        selected_block = self._selected_block_index()
        executed_blocks = selected_block + 1
        trainable_blocks = max(0, min(self.num_trainable_blocks, executed_blocks))
        trainable_start = executed_blocks - trainable_blocks
        return selected_block, trainable_start

    def train(self, mode=True):
        if self._is_fully_frozen():
            return super().train(False)
        result = super().train(mode)
        selected_block, trainable_start = self._execution_bounds()

        for block_index, block in enumerate(self.model.blocks):
            if block_index > selected_block:
                block.eval()
            elif block_index < trainable_start:
                block.eval()
            else:
                block.train(mode)

        if self.norm_layer:
            self.model.norm.train(mode)
        else:
            self.model.norm.eval()
        return result

    def _configure_trainable_parameters(self):
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        if self._is_fully_frozen():
            return

        selected_block, trainable_start = self._execution_bounds()
        for block in self.model.blocks[trainable_start : selected_block + 1]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        if self.norm_layer:
            for parameter in self.model.norm.parameters():
                parameter.requires_grad = True

    def _forward_blocks(self, tokens):
        selected_block, trainable_start = self._execution_bounds()
        if selected_block >= self.num_blocks:
            raise ValueError(
                f"Invalid output_layer {self.output_layer}; expected -1 or a value in [0, {self.num_blocks - 1}]"
            )

        if self._is_fully_frozen():
            with torch.no_grad():
                for block_index, block in enumerate(self.model.blocks):
                    tokens = block(tokens)
                    if block_index == selected_block:
                        break
            return tokens.detach()

        for block_index, block in enumerate(self.model.blocks):
            if block_index < trainable_start:
                with torch.no_grad():
                    tokens = block(tokens)
                tokens = tokens.detach()
            else:
                tokens = block(tokens)
            if block_index == selected_block:
                break
        return tokens

    def _reshape_patch_tokens(self, patch_tokens, input_shape):
        batch_size, _, height, width = input_shape
        hp, wp = height // self.patch_size, width // self.patch_size
        if patch_tokens.shape[1] != hp * wp:
            raise ValueError("Cannot reshape patch tokens into a valid spatial map")
        feature_map = patch_tokens.reshape(batch_size, hp, wp, patch_tokens.shape[-1]).permute(0, 3, 1, 2).contiguous()
        return feature_map, (hp, wp)

    def _check_finite(self, tensor, name):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values")

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("DINOv2IntermediateEncoder expects input shaped [B, 3, H, W]")

        tokens = self.model.prepare_tokens_with_masks(x)
        if tokens.ndim != 3:
            raise ValueError("prepare_tokens_with_masks must return tokens shaped [B, N + 1, C]")
        if tokens.shape[-1] != self.num_channels:
            raise ValueError("Loaded DINOv2 model returned unexpected channel dimension")
        if tokens.shape[1] < 2:
            raise ValueError("prepare_tokens_with_masks must return at least one CLS token and one patch token")

        tokens = self._forward_blocks(tokens)
        if self.norm_layer:
            tokens = self.model.norm(tokens)

        global_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        feature_map, spatial_shape = self._reshape_patch_tokens(patch_tokens, x.shape)

        self._check_finite(patch_tokens, "patch_tokens")
        self._check_finite(feature_map, "feature_map")
        self._check_finite(global_token, "global_token")

        return {
            "patch_tokens": patch_tokens,
            "feature_map": feature_map,
            "global_token": global_token,
            "spatial_shape": spatial_shape,
        }
