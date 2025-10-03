## geogfm: Build-your-own Geospatial Foundation Model (student guide)

This guide helps you build a small but complete Geospatial Foundation Model (GFM) using simple, readable components. You will implement core parts from scratch to understand how GFMs work, while using PyTorch DataLoader/optimizers and other standard tools for reliability.

What lives outside `geogfm`:
- `tests/`: unit tests for your components
- `configs/`: minimal run configs (YAML/JSON)
- `data/`: datasets and helper scripts (see `data/README.md`)

## Minimal structure you’ll use in this course

```text
geogfm/
  __init__.py
  README.md

  core/
    __init__.py
    config.py                # Minimal typed configs for model, data, training

  data/
    __init__.py
    loaders.py               # Builders: build_dataloader(...), build_datamodule(...)
    # (Samplers come later in Phase 2 if needed)
    datasets/
      __init__.py
      stac_dataset.py        # Generic STAC-backed spatiotemporal dataset
      sentinel2.py           # Example Sentinel-2 dataset
      landsat.py             # Example Landsat dataset
      climate_grid.py        # Example gridded climate dataset
    transforms/
      __init__.py
      normalization.py       # Channelwise stats, percentiles, dynamic range
      # (Augmentations optional; start simple)
      patchify.py            # Patch extraction, tiling/windowing
      temporal_sampling.py   # Frame sampling strategies (e.g., strides, seasons)
    # (Tokenizers are advanced; not needed for MVP)

  modules/
    __init__.py
    attention/
      __init__.py
      multihead_attention.py     # Standard MHA (from scratch)
    embeddings/
      __init__.py
      patch_embedding.py         # PatchEmbed(conv), linear projections
      positional_encoding.py     # Sinusoidal/Fourier/learned
      # (Temporal/geo encodings come later if needed)
    blocks/
      __init__.py
      transformer_block.py       # PreNorm/PostNorm blocks, stochastic depth
      mlp.py                     # MLP head/feedforward
      # (LayerNorm variants/residual helpers can be added later)
    losses/
      __init__.py
      mae_loss.py                # Masked reconstruction loss (MVP)
    heads/
      __init__.py
      reconstruction_head.py     # MAE decoder/readout heads
      classification_head.py
      segmentation_head.py
      regression_head.py
    # (Adapters like LoRA/prompting are out of scope for MVP)

  models/
    __init__.py
    gfm_vit.py                   # GeoViT backbone (encoder)
    gfm_mae.py                   # MAE wrapper (encoder+decoder, masking)
    # (Interop + model hub are covered later or via Prithvi)

  # (Tasks are optional; we focus on MAE pretraining.)

  training/
    __init__.py
    loop.py                      # train_step, eval_step, fit(...)
    optimizer.py                 # build_optimizer(...)
    # (Schedulers/AMP/checkpointing extras come in Phase 2)

  evaluation/
    __init__.py
    metrics.py                   # Top-k, mIoU, RMSE, R^2, retrieval metrics
    probes.py                    # Linear probes, logistic regression
    visualization.py             # Reconstructions, embeddings (UMAP/TSNE)
    nearest_neighbors.py         # Embedding kNN for retrieval/quality

  # (Inference utilities are later week topics)

  # (Interoperability with HF/timm/TorchGeo is introduced later)

  utils/
    __init__.py
    # (Add utilities as needed during the course)
```

## What you’ll build (MVP)
- Data: `stac_dataset.py`, `normalization.py`, `patchify.py`, `loaders.py`
- Modules: `patch_embedding.py`, `positional_encoding.py`, `multihead_attention.py`, `transformer_block.py`, `reconstruction_head.py`, `mae_loss.py`
- Models: `gfm_vit.py`, `gfm_mae.py`
- Training: `optimizer.py`, `loop.py`
- Evaluation: `visualization.py`

## Using standard libraries vs from-scratch
- Use PyTorch for Dataset/DataLoader, AdamW, lr schedulers, AMP, checkpointing.
- Implement core blocks from scratch first to learn: PatchEmbedding, MHA, TransformerBlock, MAE loss/head.
- Later weeks show drop-in replacements (e.g., timm ViT blocks, torch.nn.MultiheadAttention, FlashAttention) and TorchGeo datasets.

## Quick start: minimal usage

```python
from geogfm.core.config import ModelConfig, DataConfig, TrainConfig
from geogfm.data.loaders import build_dataloader
from geogfm.training.loop import fit

model_cfg = ModelConfig(architecture="gfm_vit", embed_dim=768, depth=12, image_size=224)
data_cfg = DataConfig(dataset="stac", patch_size=16, num_workers=8)
train_cfg = TrainConfig(epochs=50, batch_size=64, optimizer={"name": "adamw", "lr": 2e-4})

from geogfm.models.gfm_vit import GeoViTBackbone
from geogfm.models.gfm_mae import MaskedAutoencoder

encoder = GeoViTBackbone(model_cfg)
model = MaskedAutoencoder(model_cfg, encoder)
train_dl, val_dl = build_dataloader(data_cfg)
fit(model, (train_dl, val_dl), train_cfg)
```

## Extend as you go
- Add new datasets under `data/datasets/` and simple transforms in `data/transforms/`.
- Create new attention blocks or heads in `modules/` and try them in `gfm_vit.py`.
- Add schedulers/AMP/checkpoint helpers in `training/` when you need them.

## Conventions
- Clear names; function-first builders; slim `__init__.py` exports.
- Short docstrings with usage hints; predictable outputs for easy testing.

## Course weeks: where to look
- Week 1: data (`data/datasets`, `data/transforms`, `data/loaders`, `core/config.py`)
- Week 2: attention/embeddings/blocks (`modules/`)
- Week 3: architecture (`models/gfm_vit.py`, `modules/heads/...`)
- Week 4: MAE (`models/gfm_mae.py`, `modules/losses/mae_loss.py`)
- Week 5: training (`training/optimizer.py`, `training/loop.py`)
- Week 6: viz/metrics (`evaluation/visualization.py`)
- Weeks 7–10: interop, tasks, inference with larger models (e.g., Prithvi)



## Implementation tiers (course-oriented MVP → scale)

### MVP (essential to train a small GFM with MAE)
- geogfm/core
  - `config.py`: minimal typed configs for model, data, training.
- geogfm/data
  - `datasets/stac_dataset.py`: single dataset yielding tensors and timestamps.
  - `transforms/normalization.py`: per-channel normalization.
  - `transforms/patchify.py`: fixed-size patch extraction.
  - `loaders.py`: `build_dataloader(data_cfg) -> (train_dl, val_dl)`.
- geogfm/modules
  - `embeddings/patch_embedding.py`: conv patch embed.
  - `embeddings/positional_encoding.py`: simple positional encoding.
  - `attention/multihead_attention.py`: standard MHA.
  - `blocks/transformer_block.py`: PreNorm block (MHA + MLP).
  - `heads/reconstruction_head.py`: lightweight decoder/readout.
  - `losses/mae_loss.py`: masked reconstruction loss.
- geogfm/models
  - `gfm_vit.py`: GeoViT-style encoder.
  - `gfm_mae.py`: MAE wrapper (masking + encoder + head).
- geogfm/training
  - `optimizer.py`: AdamW builder.
  - `loop.py`: `fit`, `train_step`, `eval_step`, checkpointing minimal.
- geogfm/evaluation
  - `visualization.py`: show inputs vs reconstructions.

MVP minimal tree:
```text
geogfm/
  core/config.py
  data/loaders.py
  data/datasets/stac_dataset.py
  data/transforms/normalization.py
  data/transforms/patchify.py
  modules/attention/multihead_attention.py
  modules/embeddings/patch_embedding.py
  modules/embeddings/positional_encoding.py
  modules/blocks/transformer_block.py
  modules/heads/reconstruction_head.py
  modules/losses/mae_loss.py
  models/gfm_vit.py
  models/gfm_mae.py
  training/optimizer.py
  training/loop.py
  evaluation/visualization.py
```

### Phase 2 (efficiency, stability, basic extensibility)
- Mixed precision (`training/mixed_precision.py`), simple LR scheduler, checkpoint/EMA helpers.
- Samplers (`data/samplers.py`), temporal sampling transforms.
- Registry-lite for models/heads (`core/registry.py`) to simplify switching variants.
- Simple metrics (`evaluation/metrics.py`) like PSNR/SSIM for recon.

### Phase 3 (scale, interop, deployment, tasks)
- Interoperability (`interoperability/huggingface.py`, `timm.py`, `torchgeo.py`).
- Task heads (`tasks/*.py`) for classification/segmentation/change detection.
- Inference utilities (`inference/tiling.py`, `sliding_window.py`, `postprocess.py`).
- Prithvi compatibility (`models/prithvi_compat.py`) and hub (`models/hub/*`).

## What will be engineered in class (by repository area)

- geogfm (MVP focus)
  - Implement all MVP files above with a function-first orientation for builders and light `torch.nn.Module` layers for model parts.
- data/ (repo root)
  - `data/README.md`: dataset instructions and cache locations.
  - Optional scripts to fetch/precompute stats used by `normalization.py`.
- tests/ (repo root)
  - Unit tests for MVP components: dataset item shapes, patchify behavior, attention mask shapes, transformer block forward, loss decrease smoke test, training loop single-epoch run.
- configs/ (repo root)
  - Minimal config files for small local runs (YAML/JSON): `model.yaml`, `data.yaml`, `train.yaml` that map one-to-one to `core/config.py` schemas.
- Optional scripts/
  - Small CLI starters for launching local training or evaluation with the MVP pipeline.

## Week-by-week mapping to files and milestones

- Week 1 — Data Foundations
  - Files: `data/datasets/stac_dataset.py`, `data/transforms/normalization.py`, `data/transforms/patchify.py`, `data/loaders.py`, `core/config.py`.
  - Milestones: dataset yields normalized patches; loaders provide train/val.

- Week 2 — Attention Mechanisms
  - Files: `modules/attention/multihead_attention.py`, `modules/embeddings/positional_encoding.py`, `modules/embeddings/patch_embedding.py`, `modules/blocks/transformer_block.py`.
  - Milestones: forward passes verified, shapes stable, simple unit tests green.

- Week 3 — Complete GFM Architecture
  - Files: `models/gfm_vit.py`, `modules/heads/reconstruction_head.py`.
  - Milestones: encoder assembled; end-to-end forward on dummy input.

- Week 4 — Pretraining Implementation (MAE)
  - Files: `models/gfm_mae.py`, `modules/losses/mae_loss.py`.
  - Milestones: masking + reconstruction working on toy batch; loss computes.

- Week 5 — Training Loop Optimization
  - Files: `training/optimizer.py`, `training/loop.py` (+ optional scheduler/AMP).
  - Milestones: single-epoch run on small dataset; checkpoint save/restore.

- Week 6 — Evaluation & Analysis
  - Files: `evaluation/visualization.py` (+ optional `evaluation/metrics.py`).
  - Milestones: visualize reconstructions; track validation loss/PSNR.

- Week 7 — Integration with Existing Models
  - Files (optional): `core/registry.py` (light), `interoperability/huggingface.py` stubs.
  - Milestones: show how MVP components map to Prithvi-style structure; plan switch.

- Week 8 — Task-Specific Fine-tuning (optional in-house; primary via Prithvi)
  - Files: `tasks/classification.py` or `tasks/segmentation.py` (lightweight heads).
  - Milestones: demonstrate head swap on frozen encoder with a tiny dataset.

- Week 9 — Model Implementation & Deployment
  - Files: `inference/tiling.py`, `inference/sliding_window.py` (light stubs sufficient).
  - Milestones: explain scalable inference patterns; optional demo on a small scene.

- Week 10 — Project Presentations & Synthesis
  - Deliverables: students present MVP builds, analysis, and transition plans to Prithvi.

## From-scratch vs. library-backed components (MVP guidance)

Use standard deep learning tooling wherever it provides reliable, well-tested primitives; implement from scratch when it improves understanding of the core ideas.

- Data pipeline
  - Dataset: subclass `torch.utils.data.Dataset` [Library interface, custom implementation].
  - DataLoader: `torch.utils.data.DataLoader` [Library].
  - Sampler: start with default/`DistributedSampler` [Library]; optional custom `GeospatialSampler` later.
  - Transforms: `torchvision` where possible [Library] + custom `normalization.py`, `patchify.py` [Scratch]. Optional: `kornia`, `albumentations` later.
  - Optional geospatial datasets: `torchgeo` wrappers [Library], but we’ll implement a minimal `stac_dataset.py` [Scratch] first.

- Model components
  - PatchEmbedding: `modules/embeddings/patch_embedding.py` [Scratch], swappable with `timm` PatchEmbed [Library].
  - PositionalEncoding: `modules/embeddings/positional_encoding.py` [Scratch].
  - Multihead Attention: `modules/attention/multihead_attention.py` [Scratch for learning]; may compare to `torch.nn.MultiheadAttention` [Library].
  - Transformer Block: `modules/blocks/transformer_block.py` [Scratch]; can compare to `timm`/HF implementations [Library].
  - Reconstruction Head: `modules/heads/reconstruction_head.py` [Scratch].

- Objective
  - MAE masking + loss: `modules/losses/mae_loss.py` [Scratch].

- Training
  - Optimizer: `torch.optim.AdamW` via `training/optimizer.py` [Library].
  - Scheduler: `torch.optim.lr_scheduler` (e.g., CosineAnnealingLR) [Library, optional].
  - AMP: `torch.cuda.amp` autocast/GradScaler [Library, optional].
  - Checkpointing: `torch.save/torch.load` [Library].
  - Logging: `logging`/`tqdm` [Library]; optional TensorBoard/W&B later.

- Evaluation/Visualization
  - Visuals: `matplotlib` [Library] in `evaluation/visualization.py`.
  - Metrics: start with average loss [Scratch]; optional `torchmetrics` later [Library].

### Swappable optimized counterparts (after learning the internals)
- Attention speedups: FlashAttention / xFormers [Library] in place of naive attention.
- Backbones/blocks: `timm` ViT blocks, PatchEmbed, LayerNorm variants [Library].
- Augmentations: `kornia`, `albumentations` for geometric/photometric ops [Library].
- Data: `torchdata` datapipes; `torchgeo` datasets/transforms [Library].
- Distributed: `torch.distributed` DDP; `accelerate`/`lightning` (optional) [Library].

Coding policy for the class
- Prefer simple functions for builders; use `nn.Module` for layers/blocks.
- Start with from-scratch implementations in MVP to build intuition, then demonstrate drop-in replacements using the library-backed counterparts above.
