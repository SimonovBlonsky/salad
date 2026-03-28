from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

import ablations.train_ablations as train_ablations


class DummyEncoder(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=requires_grad)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        return {
            "patch_tokens": torch.zeros(batch_size, 4, 8, dtype=x.dtype, device=x.device),
            "feature_map": torch.ones(batch_size, 8, 2, 2, dtype=x.dtype, device=x.device),
            "global_token": torch.ones(batch_size, 8, dtype=x.dtype, device=x.device),
            "spatial_shape": (2, 2),
        }


class DummyAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            feature_map, global_token = inputs
            return feature_map.mean(dim=(2, 3)) + global_token
        return inputs.mean(dim=(2, 3))


class DummyVPRModel(nn.Module):
    def __init__(self, encoder_requires_grad=False):
        super().__init__()
        self.encoder = DummyEncoder(requires_grad=encoder_requires_grad)
        self.aggregator = DummyAggregator()
        self.agg_arch = "salad"

    def forward(self, x, **kwargs):
        features = self.encoder(x, **kwargs)
        return self.aggregator((features["feature_map"], features["global_token"]))


class DummyLoss:
    def __call__(self, descriptors, labels, miner_outputs=None):
        return descriptors.sum() * 0 + labels.float().sum() * 0 + torch.tensor(
            1.0, device=descriptors.device
        )


def build_dummy_args():
    return train_ablations.parse_args([])


def test_default_aggregator_ckpt_path_resolves_to_parent_salad_weights_from_worktree():
    expected = (
        Path(train_ablations.__file__).resolve().parents[3]
        / "weights"
        / "dino_salad_512_32.ckpt"
    )

    resolved = Path(train_ablations._default_aggregator_ckpt_path())

    assert resolved == expected


def test_parse_args_defaults_match_fair_control_experiment():
    args = train_ablations.parse_args([])

    assert args.backbone_arch == "dinov2_vitb14"
    assert args.backbone_layer == 5
    assert args.freeze_backbone is True
    assert args.num_trainable_blocks == 0
    assert args.train_aggregator_only is True
    assert args.agg_arch == "salad"
    assert args.agg_num_clusters == 16
    assert args.agg_cluster_dim == 32
    assert args.agg_token_dim == 32
    assert Path(args.aggregator_ckpt_path).is_absolute()
    assert args.aggregator_ckpt_path.endswith("weights/dino_salad_512_32.ckpt")


def test_direct_script_help_execution_works():
    worktree_root = Path(train_ablations.__file__).resolve().parents[1]
    script_path = worktree_root / "ablations" / "train_ablations.py"

    result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=worktree_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--backbone-layer" in result.stdout


def test_wrapper_construction_passes_fair_control_defaults_to_build_vpr_model(
    monkeypatch,
):
    recorded = {}

    def spy_build_vpr_model(**kwargs):
        recorded.update(kwargs)
        return DummyVPRModel(encoder_requires_grad=False)

    monkeypatch.setattr(train_ablations, "build_vpr_model", spy_build_vpr_model)
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    train_ablations.AblationLightningModule(build_dummy_args())

    assert recorded["model_name"] == "dinov2_vitb14"
    assert recorded["output_layer"] == 5
    assert recorded["freeze_backbone"] is True
    assert recorded["num_trainable_blocks"] == 0
    assert recorded["agg_arch"] == "salad"
    assert recorded["agg_config"] == {
        "num_channels": train_ablations.DINOV2_ARCHS["dinov2_vitb14"],
        "num_clusters": 16,
        "cluster_dim": 32,
        "token_dim": 32,
    }
    assert recorded["aggregator_ckpt_path"] == train_ablations._default_aggregator_ckpt_path()


def test_configure_optimizers_uses_only_aggregator_params_when_requested(monkeypatch):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=False),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    module = train_ablations.AblationLightningModule(build_dummy_args())

    optimizers, schedulers = module.configure_optimizers()
    optimizer = optimizers[0]
    scheduler_config = schedulers[0]
    scheduled_optimizer = scheduler_config["scheduler"].optimizer
    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert scheduled_optimizer is optimizer
    assert scheduler_config["interval"] == "step"
    assert optimizer_param_ids == {id(module.aggregator.weight)}
    assert id(module.encoder.weight) not in optimizer_param_ids


def test_wrapper_construction_succeeds_with_stubbed_vpr_model(monkeypatch, capsys):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=False),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    args = build_dummy_args()
    module = train_ablations.AblationLightningModule(args)
    output = module(torch.zeros(2, 3, 28, 28))
    captured = capsys.readouterr()

    assert module.model is not None
    assert module.encoder is module.model.encoder
    assert module.aggregator is module.model.aggregator
    assert output.shape == (2, 8)
    assert "backbone_arch=dinov2_vitb14" in captured.out
    assert "selected_layer=5 (zero-based DINO block index)" in captured.out
    assert "train_aggregator_only=True" in captured.out


def test_startup_validation_rejects_trainable_encoder_params_in_aggregator_only_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=True),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    with pytest.raises(
        ValueError, match="encoder parameters remain trainable in aggregator-only mode"
    ):
        train_ablations.AblationLightningModule(build_dummy_args())


def test_startup_validation_rejects_trainable_encoder_params_when_backbone_frozen(
    monkeypatch,
):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=True),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    args = train_ablations.parse_args(["--train-aggregator-only", "false"])

    with pytest.raises(
        ValueError, match="freeze_backbone=True requires all encoder parameters to be frozen"
    ):
        train_ablations.AblationLightningModule(args)


def test_validation_step_uses_index_zero_for_single_loader_path(monkeypatch):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=False),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    module = train_ablations.AblationLightningModule(build_dummy_args())
    module.val_outputs = [[]]

    descriptors = module.validation_step((torch.zeros(2, 3, 28, 28), None), 0, None)

    assert len(module.val_outputs[0]) == 1
    assert torch.equal(module.val_outputs[0][0], descriptors)


def test_on_validation_epoch_end_skips_empty_validation_outputs(monkeypatch, capsys):
    monkeypatch.setattr(
        train_ablations,
        "build_vpr_model",
        lambda **kwargs: DummyVPRModel(encoder_requires_grad=False),
    )
    monkeypatch.setattr(train_ablations.utils, "get_loss", lambda name: DummyLoss())
    monkeypatch.setattr(train_ablations.utils, "get_miner", lambda name, margin: None)

    recall_calls = []

    def record_recalls(**kwargs):
        recall_calls.append(kwargs)
        return {1: 0.0, 5: 0.0, 10: 0.0}

    monkeypatch.setattr(train_ablations.utils, "get_validation_recalls", record_recalls)

    module = train_ablations.AblationLightningModule(build_dummy_args())
    logged_metrics = []

    def record_log(name, value, **kwargs):
        logged_metrics.append((name, value, kwargs))

    monkeypatch.setattr(module, "log", record_log)
    module._trainer = SimpleNamespace(
        datamodule=SimpleNamespace(
            val_set_names=["pitts30k_val"],
            val_datasets=[
                SimpleNamespace(
                    dbStruct=SimpleNamespace(numDb=1),
                    getPositives=lambda: [[0]],
                )
            ],
        )
    )
    module.val_outputs = [[]]

    module.on_validation_epoch_end()
    captured = capsys.readouterr()

    assert "Skipping validation set pitts30k_val because no batches were produced" in captured.out
    assert recall_calls == []
    assert logged_metrics == [
        ("pitts30k_val/R1", 0.0, {"prog_bar": False, "logger": True}),
        ("pitts30k_val/R5", 0.0, {"prog_bar": False, "logger": True}),
        ("pitts30k_val/R10", 0.0, {"prog_bar": False, "logger": True}),
    ]
    assert module.val_outputs == []


def test_build_datamodule_and_trainer_smoke(monkeypatch):
    recorded = {}

    class RecordingDataModule:
        def __init__(self, **kwargs):
            recorded["datamodule_kwargs"] = kwargs

    class RecordingTrainer:
        def __init__(self, **kwargs):
            recorded["trainer_kwargs"] = kwargs

    monkeypatch.setattr(train_ablations, "GSVCitiesDataModule", RecordingDataModule)
    monkeypatch.setattr(train_ablations.pl, "Trainer", RecordingTrainer)

    args = build_dummy_args()
    model = SimpleNamespace(encoder_arch="dinov2_vitb14_layer5")

    datamodule = train_ablations.build_datamodule(args)
    trainer = train_ablations.build_trainer(args, model)

    assert isinstance(datamodule, RecordingDataModule)
    assert isinstance(trainer, RecordingTrainer)
    assert recorded["datamodule_kwargs"]["batch_size"] == args.batch_size
    assert recorded["datamodule_kwargs"]["img_per_place"] == args.img_per_place
    assert recorded["trainer_kwargs"]["max_epochs"] == args.max_epochs
    assert recorded["trainer_kwargs"]["reload_dataloaders_every_n_epochs"] == 1
