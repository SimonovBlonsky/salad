# SALAD in LoopAnything

## What this module is

This folder vendors the official SALAD implementation from the paper **"Optimal Transport Aggregation for Visual Place Recognition"**. In the original paper, SALAD is a single-stage visual place recognition (VPR) model that combines:

1. a **fine-tuned DINOv2 backbone** for strong local and global visual tokens, and
2. a new **optimal-transport-based aggregation module** that replaces NetVLAD-style soft assignment.

Within `LoopAnything`, SALAD is not used for geometry prediction directly. It is used as a **retrieval / loop-closure descriptor extractor** inside the DA3 streaming pipeline.

## Paper summary

The paper starts from the standard VPR recipe: extract dense local features from an image, aggregate them into a global descriptor, then retrieve nearest neighbors in descriptor space. The main claim is that aggregation is the key bottleneck. NetVLAD only models feature-to-cluster assignment with a row-wise softmax, while SALAD reformulates assignment as an **optimal transport (OT)** problem.

The method has three core ideas:

- **Backbone**: use DINOv2 instead of the older ResNet-style backbones. The model does not keep DINOv2 frozen; it fine-tunes only the last few transformer blocks for VPR.
- **Assignment**: compute a feature-to-cluster score matrix with a small MLP and solve assignment with the **Sinkhorn algorithm**. This enforces both feature-to-cluster and cluster-to-feature consistency instead of only a row-wise softmax.
- **Dustbin**: add one extra "dustbin" bin so the network can discard uninformative local features such as sky, road, or dynamic content.

After assignment, SALAD aggregates local features by weighted summation, concatenates this with a projected global token, and applies L2 normalization to obtain the final descriptor.

A practical takeaway from the paper is that SALAD is meant to be **fast and single-stage**. It aims to beat both older single-stage VPR systems and some re-ranking pipelines without adding a second retrieval/refinement stage.

## How the core method maps to code

### 1. Backbone

`models/backbones/dinov2.py` wraps DINOv2. The implementation mirrors the paper closely:

- the model is loaded from `torch.hub`
- only the last `num_trainable_blocks` are trainable
- earlier blocks run under `torch.no_grad()` and are detached
- the forward pass returns:
  - a spatial feature map `f` with shape `[B, C, H/14, W/14]`
  - optionally the global token `t`

This matches the paper's idea of using DINOv2 patch tokens plus the global token for VPR.

### 2. SALAD aggregation

`models/aggregators/salad.py` is the main implementation of the paper.

Important pieces:

- `score`: a small conv MLP that predicts feature-to-cluster scores
- `dust_bin`: a single learnable scalar for the dustbin column
- `log_otp_solver()` and `get_matching_probs()`: the Sinkhorn-based optimal transport solver
- `cluster_features`: a 1x1 conv MLP that reduces token dimensionality before aggregation
- `token_features`: an MLP that projects the global token

In `SALAD.forward()`:

1. local features are reduced in dimension
2. score logits are produced per spatial token
3. Sinkhorn computes a soft transport plan over clusters plus dustbin
4. the dustbin column is removed
5. local features are aggregated with weighted summation
6. the aggregated local descriptor is concatenated with the projected global token
7. the final descriptor is L2-normalized

One implementation detail worth noting: the code uses **3 Sinkhorn iterations** in the forward pass, even though the generic OT helper supports more.

### 3. Full VPR model wrapper

`vpr_model.py` defines `VPRModel`, which simply composes:

- a backbone from `models/helper.py`
- an aggregator from `models/helper.py`

The paper focuses on `DINOv2 + SALAD`, but the codebase is intentionally more general and can also instantiate ResNet, GeM, ConvAP, MixVPR, and others for ablations.

### 4. Training and evaluation scripts

- `main.py` is the main training entrypoint.
- `eval.py` is the standalone retrieval benchmark entrypoint.
- `hubconf.py` exposes a Torch Hub loader for pretrained `dinov2_salad`.

The default training configuration in `main.py` matches the paper well:

- backbone: `dinov2_vitb14`
- trainable blocks: `4`
- aggregation: `SALAD`
- clusters: `64`
- local cluster dim: `128`
- global token dim: `256`
- optimizer: `AdamW`
- loss: `MultiSimilarityLoss`
- training data: GSV-Cities through `GSVCitiesDataModule`
- validation sets: Pittsburgh30k val/test and MSLS val

`eval.py` performs standard descriptor extraction plus FAISS retrieval and reports Recall@K.

## How LoopAnything actually uses SALAD

The most important integration point is **not** inside the vendored SALAD folder itself. It is in:

- `../loop_detector.py`

That file creates a lightweight inference-only `VPRModel` with the same DINOv2-B + SALAD configuration, loads the checkpoint specified in the DA3 streaming config, extracts descriptors for all frames in a sequence, and uses FAISS inner-product search to propose loop candidates.

The retrieval logic is simple:

1. extract one global descriptor per frame
2. search top-k neighbors with FAISS
3. keep matches above a similarity threshold
4. optionally apply a simple NMS over frame indices
5. return loop pairs to the downstream Sim(3) loop-closure pipeline

So inside `LoopAnything`, SALAD should be understood as a **place-recognition front-end for loop detection**, not as part of the DA3 geometry backbone.

## Code organization cheat sheet

- `models/aggregators/salad.py`: core SALAD logic
- `models/backbones/dinov2.py`: DINOv2 wrapper with partial fine-tuning
- `models/helper.py`: registry / factory for backbones and aggregators
- `vpr_model.py`: trainable Lightning model wrapper
- `main.py`: training recipe
- `eval.py`: standalone retrieval evaluation
- `hubconf.py`: Torch Hub entrypoint
- `utils/validation.py`: Recall@K evaluation with FAISS
- `dataloaders/`: training and benchmark dataset wrappers
- `../loop_detector.py`: the place where DA3 streaming actually calls SALAD

## Recommended reading order

If the goal is to understand the method first, read:

1. `paper/2311.15937v2.pdf`
2. `models/aggregators/salad.py`
3. `models/backbones/dinov2.py`
4. `vpr_model.py`

If the goal is to understand how `LoopAnything` uses SALAD in practice, read:

1. `../loop_detector.py`
2. `vpr_model.py`
3. `models/helper.py`
4. `models/aggregators/salad.py`

## Bottom line

SALAD is a compact VPR subsystem built around **DINOv2 features + Sinkhorn-based optimal transport aggregation**. In the standalone repository it is a trainable place-recognition model. In `LoopAnything`, it is mainly a **loop-closure retrieval engine** that supplies candidate frame pairs to the longer-range DA3 streaming pipeline.
