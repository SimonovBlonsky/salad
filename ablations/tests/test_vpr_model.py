import torch
import torch.nn as nn
import pytest

from ablations.models import vpr_helper as helper
from ablations.models.dinov2_intermediate import DINOv2IntermediateEncoder
from ablations.models.vpr_model import VPRModel
from models.aggregators import SALAD


class DummyEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x, **kwargs):
        return self.features


class RecordingAggregator(nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.calls = []

    def forward(self, inputs):
        self.calls.append(inputs)
        return self.output


class DummyDinoBackbone(nn.Module):
    def __init__(self, channels=384, num_blocks=2, patch_size=14):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(num_blocks)])
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, x):
        batch, _, height, width = x.shape
        hp = height // self.patch_size
        wp = width // self.patch_size
        tokens = torch.zeros(
            batch,
            hp * wp + 1,
            self.channels,
            dtype=x.dtype,
            device=x.device,
        )
        return tokens


def make_prefixed_state_dict(prefix, module):
    return {
        f"{prefix}{key}": value.clone() for key, value in module.state_dict().items()
    }


def test_salad_aggregator_uses_feature_map_and_global_token_and_returns_features():
    expected_patch_tokens = torch.ones(2, 4, 8)
    expected_feature_map = torch.full((2, 8, 2, 2), 2.0)
    expected_global_token = torch.full((2, 8), 3.0)
    expected_spatial_shape = (2, 2)
    features = {
        "patch_tokens": expected_patch_tokens.clone(),
        "feature_map": expected_feature_map.clone(),
        "global_token": expected_global_token.clone(),
        "spatial_shape": expected_spatial_shape,
    }
    descriptor = torch.full((2, 8), 5.0)
    aggregator = RecordingAggregator(descriptor)
    model = VPRModel(DummyEncoder(features), aggregator, "SaLaD")

    output, returned_features = model(torch.zeros(2, 3, 8, 8), return_features=True)

    assert model.agg_arch == "salad"
    assert output is descriptor
    assert torch.equal(returned_features["patch_tokens"], expected_patch_tokens)
    assert torch.equal(returned_features["feature_map"], expected_feature_map)
    assert torch.equal(returned_features["global_token"], expected_global_token)
    assert returned_features["spatial_shape"] == expected_spatial_shape
    assert len(aggregator.calls) == 1
    assert isinstance(aggregator.calls[0], tuple)
    assert torch.equal(aggregator.calls[0][0], features["feature_map"])
    assert torch.equal(aggregator.calls[0][1], features["global_token"])


def test_non_salad_aggregator_uses_feature_map_only():
    features = {
        "patch_tokens": torch.ones(1, 2, 4),
        "feature_map": torch.full((1, 4, 1, 2), 7.0),
        "global_token": torch.full((1, 4), 11.0),
        "spatial_shape": (1, 2),
    }
    descriptor = torch.full((1, 4), 13.0)
    aggregator = RecordingAggregator(descriptor)
    model = VPRModel(DummyEncoder(features), aggregator, "avgpool")

    output = model(torch.zeros(1, 3, 4, 4))

    assert model.agg_arch == "avgpool"
    assert output is descriptor
    assert len(aggregator.calls) == 1
    assert torch.equal(aggregator.calls[0], features["feature_map"])


def test_non_finite_descriptor_raises_value_error():
    features = {
        "patch_tokens": torch.ones(1, 2, 4),
        "feature_map": torch.full((1, 4, 1, 2), 1.0),
        "global_token": torch.full((1, 4), 1.0),
        "spatial_shape": (1, 2),
    }

    class NonFiniteAggregator(nn.Module):
        def forward(self, inputs):
            return torch.tensor([float("nan")])

    model = VPRModel(DummyEncoder(features), NonFiniteAggregator(), "salad")

    with pytest.raises(
        ValueError, match="Non-finite descriptor produced by ablation VPRModel"
    ):
        model(torch.zeros(1, 3, 4, 4))


def test_build_salad_aggregator_returns_salad_instance_with_expected_dims():
    aggregator = helper.build_aggregator(
        "SALAD",
        {
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
    )

    assert isinstance(aggregator, SALAD)
    assert aggregator.num_channels == 384
    assert aggregator.num_clusters == 4
    assert aggregator.cluster_dim == 8
    assert aggregator.token_dim == 16


def test_extract_prefixed_state_dict_strips_matching_prefixes():
    state_dict = {
        "model.aggregator.dust_bin": torch.tensor(1.5),
        "module.aggregator.cluster_features.0.weight": torch.ones(2, 2),
        "encoder.weight": torch.zeros(1),
    }

    extracted = helper.extract_prefixed_state_dict(
        state_dict, helper.AGGREGATOR_STATE_PREFIXES
    )

    assert set(extracted) == {"dust_bin", "cluster_features.0.weight"}
    assert torch.equal(extracted["dust_bin"], torch.tensor(1.5))
    assert torch.equal(extracted["cluster_features.0.weight"], torch.ones(2, 2))
    assert "encoder.weight" not in extracted


def test_load_aggregator_weights_from_salad_ckpt_updates_aggregator_and_ignores_non_agg_keys(
    tmp_path,
):
    aggregator = helper.build_aggregator(
        "salad",
        {
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
    )
    ckpt_path = tmp_path / "salad_checkpoint.pt"
    torch.save(
        {
            "model": {
                "aggregator.dust_bin": torch.tensor(2.5),
                "encoder.weight": torch.tensor(9.0),
            }
        },
        ckpt_path,
    )

    with pytest.warns(
        UserWarning, match="partial/incomplete aggregator checkpoint load"
    ):
        helper.load_aggregator_weights_from_salad_ckpt(
            aggregator, ckpt_path, strict=False
        )

    assert aggregator.dust_bin.item() == pytest.approx(2.5)


def test_load_aggregator_weights_from_salad_ckpt_non_strict_rejects_wrong_aggregator(
    tmp_path,
):
    target = helper.build_aggregator(
        "convap",
        {
            "in_channels": 384,
            "out_channels": 16,
            "s1": 2,
            "s2": 2,
        },
    )
    source = helper.build_aggregator(
        "salad",
        {
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
    )
    ckpt_path = tmp_path / "salad_checkpoint.pt"
    torch.save(
        {"state_dict": make_prefixed_state_dict("aggregator.", source)}, ckpt_path
    )

    with pytest.raises(ValueError, match="incompatible"):
        helper.load_aggregator_weights_from_salad_ckpt(target, ckpt_path, strict=False)


def test_load_aggregator_weights_from_salad_ckpt_strict_mode_fails_on_partial_checkpoint(
    tmp_path,
):
    aggregator = helper.build_aggregator(
        "salad",
        {
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
    )
    ckpt_path = tmp_path / "salad_partial_checkpoint.pt"
    torch.save({"aggregator.dust_bin": torch.tensor(2.5)}, ckpt_path)

    with pytest.raises(RuntimeError, match="Missing key\\(s\\) in state_dict"):
        helper.load_aggregator_weights_from_salad_ckpt(
            aggregator, ckpt_path, strict=True
        )


def test_build_vpr_model_assembles_dinov2_encoder_and_returns_eval_model(
    tmp_path, monkeypatch
):
    backbone = DummyDinoBackbone(channels=384)

    def hub_loader(repo, model_name):
        assert repo == "facebookresearch/dinov2"
        assert model_name == "dinov2_vits14"
        return backbone

    real_build_dino_encoder = helper.build_dino_encoder
    build_calls = {}

    def spy_build_dino_encoder(**kwargs):
        build_calls.update(kwargs)
        return real_build_dino_encoder(**kwargs)

    monkeypatch.setattr(helper, "build_dino_encoder", spy_build_dino_encoder)

    source = helper.build_aggregator(
        "salad",
        {
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
    )
    ckpt_path = tmp_path / "aggregator_checkpoint.pt"
    torch.save(
        {"state_dict": make_prefixed_state_dict("aggregator.", source)}, ckpt_path
    )

    model = helper.build_vpr_model(
        model_name="dinov2_vits14",
        hub_loader=hub_loader,
        agg_arch="salad",
        agg_config={
            "num_channels": 384,
            "num_clusters": 4,
            "cluster_dim": 8,
            "token_dim": 16,
            "dropout": 0.0,
        },
        aggregator_ckpt_path=ckpt_path,
    )

    assert isinstance(model, VPRModel)
    assert isinstance(model.encoder, DINOv2IntermediateEncoder)
    assert model.encoder.model is backbone
    assert isinstance(model.aggregator, SALAD)
    assert build_calls["model_name"] == "dinov2_vits14"
    assert not model.training


def test_build_vpr_model_fails_before_encoder_construction_on_invalid_aggregator(
    monkeypatch,
):
    def fail_if_called(**kwargs):
        raise AssertionError("encoder should not be constructed")

    monkeypatch.setattr(helper, "build_dino_encoder", fail_if_called)

    with pytest.raises(ValueError, match="Unsupported aggregator"):
        helper.build_vpr_model(
            model_name="dinov2_vits14",
            hub_loader=lambda *args, **kwargs: None,
            agg_arch="not_a_real_aggregator",
            agg_config={},
        )


def test_build_vpr_model_fails_before_encoder_construction_on_incompatible_checkpoint(
    tmp_path, monkeypatch
):
    def fail_if_called(**kwargs):
        raise AssertionError("encoder should not be constructed")

    monkeypatch.setattr(helper, "build_dino_encoder", fail_if_called)

    wrong_aggregator = helper.build_aggregator(
        "convap",
        {
            "in_channels": 384,
            "out_channels": 16,
            "s1": 2,
            "s2": 2,
        },
    )
    ckpt_path = tmp_path / "wrong_aggregator_checkpoint.pt"
    torch.save(
        {"state_dict": make_prefixed_state_dict("aggregator.", wrong_aggregator)},
        ckpt_path,
    )

    with pytest.raises(ValueError, match="incompatible"):
        helper.build_vpr_model(
            model_name="dinov2_vits14",
            hub_loader=lambda *args, **kwargs: None,
            agg_arch="salad",
            agg_config={
                "num_channels": 384,
                "num_clusters": 4,
                "cluster_dim": 8,
                "token_dim": 16,
                "dropout": 0.0,
            },
            aggregator_ckpt_path=ckpt_path,
            strict=False,
        )
