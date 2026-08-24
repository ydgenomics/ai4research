# Modularization plan and process log

This is the single source of truth for all modularization work on the training and inference pipelines. **Update this file whenever the layout, behavior, or status changes.**

---

## Phase 1 — Training modularization

### Goals

- Extract shared training logic into the `model/` package.
- Add a unified `training.py` entrypoint; YAML `predictor.type` selects the head (`v0`, `wtd`, `h2`, `h2b`).
- Legacy monolithic trainers archived under `model.bak/` (`training.v0.py`, `training.H2.py`); not updated with `model/` changes.
- Snapshot the original training config as `config.example/training.v0.yaml`; new `config.example/training.yaml` has a `predictor:` section.

### Target layout

| Path | Role |
|------|------|
| `model/env.py` | Env vars, logging, `SEED`, seeding |
| `model/distributed.py` | DDP helpers, `DistributedSamplerCallback`, SyncBatchNorm |
| `model/scaling.py` | `targets_scaling` |
| `model/index.py` | `build_index` |
| `model/dataset.py` | LEGACY shim (do not edit; prefer `model/dataset_strand.py`) |
| `model/dataset_strand.py` | canonical `LazyGenomicDataset` + `InferenceDataset` |
| `model/encoder.py` | `ATAC_Encoder` |
| `model/predictor_v0.py` | MSE `MultiModalPredictor` |
| `model/predictor_wtd.py` | weighted-MSE `MultiModalPredictorWTD` |
| `model/predictor_h2.py` | H2 `MultiModalPredictorH2` |
| `model/predictor_h2b.py` | H2B `MultiModalPredictorH2B` (current) |
| `model/trainer.py` | `CustomTrainer` (v0); `CustomTrainerH2` extends it |
| `model/load_pretrained.py` | `load_model_and_tokenizer` |
| `model/pipeline.py` | `run_training`, `run_training_from_yaml` |
| `training.py` | CLI: `python training.py <config.yaml>` |

### YAML: predictor selection

```yaml
predictor:
  type: v0   # original MSE head (default if `predictor` omitted)
  # type: wtd  # weighted MSE head
  # type: h2   # BCE + masked MSE regression head (non-zero positions)
  # type: h2b  # H2B head (current: model/predictor_h2b.py)
  # lambda_reg: 1.0
  # classification_pos_weight: 10.0
```

For `h2`, optional `lambda_reg` and `classification_pos_weight` under `predictor:` (legacy: under `training:`). Modular `predictor_h2` uses masked **MSE** on non-zero labels; archived `model.bak/training.H2.py` used masked **L1** for the same role.
For `h2b`, the `model/predictor_h2b.v1.py` and `model/predictor_h2b.v2.py` files are **legacy snapshots**; use `model/predictor_h2b.py` for current runs.

### Todo checklist

- [x] `model/env.py`, `model/distributed.py`, `model/scaling.py`
- [x] `model/index.py`, `model/dataset.py`
- [x] `model/encoder.py`, `model/predictor_v0.py`, `model/predictor_h2.py`
- [x] `model/trainer.py`, `model/load_pretrained.py`
- [x] `model/pipeline.py` + root `training.py`
- [x] `config.example/training.v0.yaml` (snapshot) + new `config.example/training.yaml` with `predictor:` block
- [x] Import / CLI smoke check (`python -c "import model.pipeline"`, `python training.py --help`)
- [x] `AGENTS.md` updated (structure, config snippet, commands, DDP example, checkpoint note)

---

## Phase 2 — Inference modularization

### Goals

- Restructure `inference.py` to import shared code from `model/` instead of duplicating it.
- Eliminate risk of silent architectural drift between training and inference.
- Original `inference.py` snapshot lives as `model.bak/inference.v0.py`; `config.example/infer.yaml` snapshot is `config.example/infer.v0.yaml`.
- Keep the public CLI unchanged: `python inference.py <config>.yaml`.

### Gap analysis — what `inference.py` duplicates vs `model/`

| Component | `inference.py` (current) | `model/` | Action |
|---|---|---|---|
| `targets_scaling` | local copy | `model/scaling.py` | Remove local; import from `model.scaling` |
| `predictions_scaling` | local only | missing | Add to `model/scaling.py`; import |
| `ATAC_Encoder` | local copy, default `output_dim=4096`, no `.float()` cast | `model/encoder.py`, default 1024, casts to float32 | Remove local; import from `model.encoder` (both called as `ATAC_Encoder(1024)`) |
| `MultiModalPredictor` (v0) | local copy, all layers run under `no_grad` in forward | `model/predictor_v0.py`, last layer outside `no_grad` | Remove local; import from `model.predictor_v0` (call site wraps `with torch.no_grad()`, so equivalent) |
| `InferenceDataset` | local, returns extra metadata fields | `model/dataset_strand.py` provides `InferenceDataset` with metadata | Import from `model/dataset_strand.py` |
| `build_index` | local, takes `cfg` dict in infer-YAML format, no CSV caching | `model/index.py`, takes explicit args, saves CSV | Replace with thin `_parse_infer_cfg` adapter + call to `model.index.build_index` |
| model / tokenizer loading | inline, hardcodes `bfloat16` | `model/load_pretrained.py`, reads `model_torch_dtype` from cfg | Import `load_model_and_tokenizer`; add `model_torch_dtype: bfloat16` to `infer.yaml` |
| `save_result` | inference-specific output logic | not in model | Keep in `inference.py` |
| `main()` | inference orchestration | not in model | Keep in `inference.py` |

### Additions to `model/`

#### `model/scaling.py` — add `predictions_scaling`

```python
def predictions_scaling(predictions, track_means, apply_squashing=True):
    predictions = np.where(predictions > 10.0, (predictions + 10.0) ** 2 / (4 * 10.0), predictions)
    if apply_squashing:
        predictions = predictions ** (1.0 / 0.75)
    predictions = predictions * track_means[:, None]
    return np.nan_to_num(predictions, nan=0.0)
```

#### `model/dataset_strand.py` — add `InferenceDataset`

Same BigWig/FASTA loading as `LazyGenomicDataset` but `__getitem__` also returns:

- `track_mean_plus`, `track_mean_minus` — for unscaling predictions
- `sequence` — for slicing output to true sequence length
- `position` — `(chrom, start, end)` for output records
- `batch_name_plus`, `batch_name_minus` — for partitioning output files

### Changes to `inference.py`

Remove local class/function definitions and replace with imports:

```python
from model.scaling       import targets_scaling, predictions_scaling
from model.encoder       import ATAC_Encoder
from model.predictor_v0  import MultiModalPredictor
from model.dataset_strand import InferenceDataset
from model.load_pretrained import load_model_and_tokenizer
from model.index         import build_index as _build_index
```

Training and inference share `model.dataset_config.parse_dataset_block(cfg)`: structured rows under **`training_data`** (training YAML) or **`test_data`** (infer YAML), optional per-entry `chromosome`, plus legacy string `cell_types` + `rna_files`/`atac_files`. In `inference.py` / `pipeline.py`:

```python
from model.dataset_config import parse_dataset_block

rna_files, atac_files, cell_types, chromosome, chromosome_per_cell_type = (
    parse_dataset_block(cfg)
)
index_df = _build_index(
    ...,
    rna_files,
    atac_files,
    cell_types,
    chromosome,
    chromosome_per_cell_type=chromosome_per_cell_type,
)
```

### Config update — `config.example/infer.yaml`

Add one key so `load_model_and_tokenizer` sets dtype correctly:

```yaml
model_torch_dtype: "bfloat16"
```

### Legacy backups

| Original | Backup |
|---|---|
| `inference.py` | `inference.v0.py` |
| `config.example/infer.yaml` | `config.example/infer.v0.yaml` |

### Todo checklist

- [x] Copy `inference.py` → `inference.v0.py`
- [x] Copy `config.example/infer.yaml` → `config.example/infer.v0.yaml`
- [x] Add `predictions_scaling` to `model/scaling.py`
- [x] Add `InferenceDataset` to `model/dataset.py`
- [x] Rewrite `inference.py`: remove duplicates, add imports, replace `build_index`, replace model-loading
- [x] Add `model_torch_dtype: bfloat16` and `atac_encoder_output_dim` to `config.example/infer.yaml`
- [x] Import smoke check: all `model/` components imported; `inference.py` AST-parsed successfully
- [x] Update `AGENTS.md` inference section to reflect new structure

---

## Phase 3 — Strand-split dataset and `predictor.type: ss`

### Goals

- Train and infer with **one strand per sample** (`labels` shape `[1, L]`) instead of two channels per window (`[2, L]`).
- Keep the default **`dataset_type: v0`** path on `model/dataset.py`; optional **`dataset_type: strand`** uses `model/dataset_strand.py`.
- Add **`MultiModalPredictorSS`** (`model/predictor_ss.py`) with `REQUIRED_DATASET_TYPE = "strand"` and validate against config via `_validate_dataset_type_match` in `model/pipeline.py` (also used by `inference.py`).
- Forward **`dataset_type`** from experiment/training config into infer YAML in `run.py`.

### Layout additions

| Path | Role |
|------|------|
| `model/dataset_strand.py` | `LazyGenomicDataset` / `InferenceDataset`: `__len__ = 2 × rows`, strand metadata for inference |
| `model/predictor_ss.py` | `MultiModalPredictorSS`: same backbone as v0, 1-channel head |

### Todo checklist

- [x] `model/dataset_strand.py` — base helpers + doubled train/infer datasets
- [x] `model/predictor_ss.py` — `REQUIRED_DATASET_TYPE`, 1-channel final conv
- [x] `model/pipeline.py` — `ss` in `build_multimodal_model`, `_resolve_dataset_type`, dataset import switch
- [x] `inference.py` — `dataset_type` branch and strand output loop
- [x] `run.py` — pass `dataset_type` into infer config
- [x] `config.example/*.yaml` — comments for `dataset_type` / `ss`
- [x] `AGENTS.md`, `README.md`, `training.py` docstring — document strand-split and `ss`

---

## Process log

| When | What | Phase |
|------|------|-------|
| Init | Created doc; Phase 1 goals and layout defined | 1 |
| Step 1 | Added `model/__init__.py`, `env.py`, `distributed.py`, `scaling.py` | 1 |
| Step 2 | Added `index.py`, `dataset.py` | 1 |
| Step 3 | Added `encoder.py`, `predictor_v0.py`, `predictor_h2.py` | 1 |
| Step 4 | Added `trainer.py`, `load_pretrained.py` | 1 |
| Step 5 | Added `pipeline.py` with predictor factory; added root `training.py` | 1 |
| Step 6 | Snapshotted `config.example/training.v0.yaml`; rewrote `config.example/training.yaml` | 1 |
| Step 7 | Smoke checks passed; `AGENTS.md` updated | 1 |
| Step 8 | Phase 2 plan merged into this doc; inference backups and todo list added | 2 |
| Step 9 | Backed up `inference.py` → `inference.v0.py`, `infer.yaml` → `infer.v0.yaml` | 2 |
| Step 10 | Added `predictions_scaling` to `model/scaling.py` | 2 |
| Step 11 | Added `InferenceDataset` + `_process_signal` helper to `model/dataset.py` | 2 |
| Step 12 | Rewrote `inference.py`: removed all duplicated classes/functions; imports from `model/`; `_parse_infer_cfg` adapter for infer-YAML format; `save_result` and `main()` kept | 2 |
| Step 13 | Updated `config.example/infer.yaml`: added `model_torch_dtype`, `atac_encoder_output_dim` | 2 |
| Step 14 | Smoke checks passed: all `model/` imports OK, `inference.py` syntax clean | 2 |
| Step 15 | Updated `AGENTS.md`: tree (inference.v0.py, infer.v0.yaml), config snippet, `model/` note | 2 |
| Step 16 | H2 inference: `MultiModalPredictorH2` sets `logits` to `value_pred * (P(non-zero)≥0.5)` when `labels is None`; `inference.py` uses `build_multimodal_model` + optional `predictor.type: h2` | 2 |
| Step 17 | Phase 3: `dataset_strand.py`, `predictor_ss.py`, `dataset_type` + `predictor.type: ss` wiring in `pipeline.py`, `inference.py`, `run.py` | 3 |
| Step 18 | Docs: `AGENTS.md`, `README.md`, `training.py`, this file — strand-split / `ss` / `REQUIRED_DATASET_TYPE` | 3 |
