from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    worktree_root = Path(__file__).resolve().parents[1]
    if str(worktree_root) not in sys.path:
        sys.path.insert(0, str(worktree_root))

import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler

import utils
from ablations.models.dinov2_intermediate import DINOV2_ARCHS
from ablations.models.vpr_helper import build_vpr_model


GSVCitiesDataModule = None


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _default_aggregator_ckpt_path():
    train_script = Path(__file__).resolve()
    worktree_root = train_script.parents[1]
    worktree_candidate = worktree_root / "weights" / "dino_salad_512_32.ckpt"
    if worktree_candidate.exists():
        return str(worktree_candidate)
    if worktree_root.parent.name == ".worktrees":
        return str(worktree_root.parent.parent / "weights" / "dino_salad_512_32.ckpt")
    return str(worktree_candidate)


class AblationLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(vars(args))

        agg_arch = args.agg_arch.lower()
        if agg_arch != "salad":
            raise ValueError(f"Unsupported aggregator architecture for this script: {args.agg_arch}")

        if args.backbone_arch not in DINOV2_ARCHS:
            raise ValueError(f"Unsupported DINOv2 backbone architecture: {args.backbone_arch}")

        self.encoder_arch = f"{args.backbone_arch}_layer{args.backbone_layer}"
        self.loss_fn = utils.get_loss(args.loss_name)
        self.miner = utils.get_miner(args.miner_name, args.miner_margin)
        self.batch_acc = []
        self.faiss_gpu = args.faiss_gpu

        self.model = build_vpr_model(
            model_name=args.backbone_arch,
            output_layer=args.backbone_layer,
            freeze_backbone=args.freeze_backbone,
            num_trainable_blocks=args.num_trainable_blocks,
            norm_layer=args.norm_layer,
            return_token=True,
            agg_arch=agg_arch,
            agg_config=self._build_aggregator_config(args),
            aggregator_ckpt_path=args.aggregator_ckpt_path,
        )
        self.encoder = self.model.encoder
        self.aggregator = self.model.aggregator
        self.val_outputs = []

        self._validate_trainability()
        self._print_startup_diagnostics()

    def _build_aggregator_config(self, args):
        return {
            "num_channels": DINOV2_ARCHS[args.backbone_arch],
            "num_clusters": args.agg_num_clusters,
            "cluster_dim": args.agg_cluster_dim,
            "token_dim": args.agg_token_dim,
        }

    def _encoder_parameters(self):
        return list(self.encoder.parameters())

    def _aggregator_parameters(self):
        return list(self.aggregator.parameters())

    def _validate_trainability(self):
        encoder_parameters = self._encoder_parameters()
        trainable_encoder_parameters = [
            parameter for parameter in encoder_parameters if parameter.requires_grad
        ]

        if self.args.train_aggregator_only and trainable_encoder_parameters:
            raise ValueError("encoder parameters remain trainable in aggregator-only mode")

        if self.args.freeze_backbone and trainable_encoder_parameters:
            raise ValueError("freeze_backbone=True requires all encoder parameters to be frozen")

    def _optimizer_parameters(self):
        if self.args.train_aggregator_only:
            parameters = [
                parameter
                for parameter in self._aggregator_parameters()
                if parameter.requires_grad
            ]
        else:
            parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]

        if not parameters:
            raise ValueError("No trainable parameters found for optimization")
        return parameters

    def _print_startup_diagnostics(self):
        total_params = sum(parameter.numel() for parameter in self.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        optimizer_params = sum(parameter.numel() for parameter in self._optimizer_parameters())
        print(f"[train_ablations] backbone_arch={self.args.backbone_arch}")
        print(
            "[train_ablations] selected_layer="
            f"{self.args.backbone_layer} (zero-based DINO block index)"
        )
        print(f"[train_ablations] freeze_backbone={self.args.freeze_backbone}")
        print(f"[train_ablations] num_trainable_blocks={self.args.num_trainable_blocks}")
        print(f"[train_ablations] train_aggregator_only={self.args.train_aggregator_only}")
        print(f"[train_ablations] total_params={total_params}")
        print(f"[train_ablations] trainable_params={trainable_params}")
        print(f"[train_ablations] optimizer_param_count={optimizer_params}")

    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)

    def _loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log("b_acc", sum(self.batch_acc) / len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        batch_size, images_per_place, channels, height, width = places.shape
        images = places.view(batch_size * images_per_place, channels, height, width)
        labels = labels.view(-1)

        descriptors = self(images)
        if not torch.isfinite(descriptors).all():
            raise ValueError("Non-finite descriptors produced during training")

        loss = self._loss_function(descriptors, labels)
        self.log("loss", loss.item(), logger=True, prog_bar=True)
        return {"loss": loss}

    def on_train_epoch_end(self):
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        target_index = 0 if dataloader_idx is None else dataloader_idx
        self.val_outputs[target_index].append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_start(self):
        self.val_outputs = [[] for _ in range(len(self.trainer.datamodule.val_datasets))]

    def on_validation_epoch_end(self):
        datamodule = self.trainer.datamodule
        for index, (val_set_name, val_dataset) in enumerate(
            zip(datamodule.val_set_names, datamodule.val_datasets)
        ):
            if index >= len(self.val_outputs) or not self.val_outputs[index]:
                self.log(f"{val_set_name}/R1", 0.0, prog_bar=False, logger=True)
                self.log(f"{val_set_name}/R5", 0.0, prog_bar=False, logger=True)
                self.log(f"{val_set_name}/R10", 0.0, prog_bar=False, logger=True)
                print(
                    f"[train_ablations] Skipping validation set {val_set_name} "
                    "because no batches were produced"
                )
                continue

            features = torch.concat(self.val_outputs[index], dim=0)

            if "pitts" in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                positives = val_dataset.getPositives()
            elif "msls" in val_set_name:
                num_references = val_dataset.num_references
                positives = val_dataset.pIdx
            else:
                raise NotImplementedError(
                    f"Please implement validation handling for {val_set_name}"
                )

            reference_features = features[:num_references]
            query_features = features[num_references:]
            recalls = utils.get_validation_recalls(
                r_list=reference_features,
                q_list=query_features,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
            )

            self.log(f"{val_set_name}/R1", recalls[1], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R5", recalls[5], prog_bar=False, logger=True)
            self.log(f"{val_set_name}/R10", recalls[10], prog_bar=False, logger=True)

        print("\n\n")
        self.val_outputs = []

    def configure_optimizers(self):
        trainable_parameters = self._optimizer_parameters()

        optimizer_name = self.args.optimizer.lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=self.args.momentum,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                trainable_parameters,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")

        lr_sched_name = self.args.lr_sched.lower()
        if lr_sched_name == "multistep":
            scheduler = lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.args.lr_sched_milestones,
                gamma=self.args.lr_sched_gamma,
            )
        elif lr_sched_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.lr_sched_tmax,
            )
        elif lr_sched_name == "linear":
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.args.lr_sched_start_factor,
                end_factor=self.args.lr_sched_end_factor,
                total_iters=self.args.lr_sched_total_iters,
            )
        else:
            raise ValueError(f"Unsupported lr scheduler: {self.args.lr_sched}")

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train SALAD ablations with Lightning.")

    parser.add_argument("--backbone-arch", default="dinov2_vitb14")
    parser.add_argument(
        "--backbone-layer",
        type=int,
        default=5,
        help="Zero-based DINO block index to extract features from.",
    )
    parser.add_argument("--freeze-backbone", type=_str_to_bool, default=True)
    parser.add_argument("--num-trainable-blocks", type=int, default=0)
    parser.add_argument("--norm-layer", type=_str_to_bool, default=True)
    parser.add_argument("--train-aggregator-only", type=_str_to_bool, default=True)

    parser.add_argument("--agg-arch", default="salad")
    parser.add_argument("--agg-num-clusters", type=int, default=16)
    parser.add_argument("--agg-cluster-dim", type=int, default=32)
    parser.add_argument("--agg-token-dim", type=int, default=32)
    parser.add_argument(
        "--aggregator-ckpt-path",
        default=_default_aggregator_ckpt_path(),
    )

    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--img-per-place", type=int, default=4)
    parser.add_argument("--min-img-per-place", type=int, default=4)
    parser.add_argument("--shuffle-all", type=_str_to_bool, default=False)
    parser.add_argument("--random-sample-from-each-place", type=_str_to_bool, default=True)
    parser.add_argument("--image-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument("--show-data-stats", type=_str_to_bool, default=True)
    parser.add_argument(
        "--val-set-names",
        nargs="+",
        default=["pitts30k_val", "pitts30k_test"],
    )

    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--weight-decay", type=float, default=9.5e-9)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-sched", default="linear")
    parser.add_argument("--lr-sched-start-factor", type=float, default=1.0)
    parser.add_argument("--lr-sched-end-factor", type=float, default=0.2)
    parser.add_argument("--lr-sched-total-iters", type=int, default=4000)
    parser.add_argument("--lr-sched-milestones", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--lr-sched-gamma", type=float, default=0.1)
    parser.add_argument("--lr-sched-tmax", type=int, default=10)

    parser.add_argument("--loss-name", default="MultiSimilarityLoss")
    parser.add_argument("--miner-name", default="MultiSimilarityMiner")
    parser.add_argument("--miner-margin", type=float, default=0.1)
    parser.add_argument("--faiss-gpu", type=_str_to_bool, default=False)

    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--default-root-dir", default="./logs/ablations")
    parser.add_argument("--precision", default="16-mixed")
    parser.add_argument("--max-epochs", type=int, default=4)
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1)
    parser.add_argument("--num-sanity-val-steps", type=int, default=0)
    parser.add_argument("--reload-dataloaders-every-n-epochs", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--fit-ckpt-path", default=None)

    return parser.parse_args(argv)


def build_datamodule(args):
    datamodule_cls = GSVCitiesDataModule
    if datamodule_cls is None:
        from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule as datamodule_cls

    return datamodule_cls(
        batch_size=args.batch_size,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        shuffle_all=args.shuffle_all,
        random_sample_from_each_place=args.random_sample_from_each_place,
        image_size=tuple(args.image_size),
        num_workers=args.num_workers,
        show_data_stats=args.show_data_stats,
        val_set_names=args.val_set_names,
    )


def build_trainer(args, model):
    monitor_name = f"{args.val_set_names[0]}/R1"
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        monitor=monitor_name,
        filename=f"{model.encoder_arch}_{{epoch:02d}}",
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        save_last=True,
        mode="max",
    )

    return pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.default_root_dir,
        num_nodes=args.num_nodes,
        num_sanity_val_steps=args.num_sanity_val_steps,
        precision=args.precision,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
    )


def main(argv=None):
    args = parse_args(argv)
    model = AblationLightningModule(args)
    datamodule = build_datamodule(args)
    trainer = build_trainer(args, model)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=args.fit_ckpt_path)
    return trainer


if __name__ == "__main__":
    main()
