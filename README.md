# Vertical Flow

Reproduction code for the Vertical Flow paper experiments.
Source files in `src/` are the canonical runnable entry points; `figures/` contains lightweight plotting wrappers.

## Repository layout

```text
VerticalFlow/
├── src/                    # Runnable source files and implementation modules
│   ├── generation.py       # Vertical-flow generation
│   ├── experiment.py       # Table 3 / Table 6 correction-method runner
│   ├── dualstream.py       # Dual-stream patching implementation
│   ├── error_decomposition.py  # Table 5 error decomposition
│   ├── mlp_probe.py        
│   ├── linear_probe.py     
│   ├── incarry_probe.py    # Incoming-carry probe analysis
│   ├── models.py           # Shared probe model definitions
│   ├── plotting/           # Plotting implementations
│   └── utils/              # Shared CLI, metrics, data, and model helpers
├── figures/                
├── data/                   # Input arithmetic datasets
└── results/               
```

## Environment

```bash
pip install -e .
pip install -r requirements.txt
```

## Vertical flow generation

```bash
python src/generation.py \
  --model-path Qwen/Qwen3-4B \
  --dataset data/num3len10-100000.pkl \
  --output-h5 results/activations/plus_num3len10_Qwen3-4B_nocheckall_balance_both.h5
```

## Table 3: Dual-stream patching and correction baselines

```bash
python src/experiment.py \
  --method dual-stream \
  --h5 results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5 \
  --dataset data/num3len10-10000.pkl \
  --model Qwen/Qwen3-4B
```

Supported correction methods are `replacement`, `steering`, `dual-stream`, `prompt`, and `all`.

## Table 5: Error decomposition

```bash
python src/error_decomposition.py \
  --h5 results/activations/plus_num3len10_Qwen3-4b/plus_num3len10_Qwen3-4b.h5 \
  --dataset data/num3len10-10000.pkl
```

Useful options:

- `--balance-mode none|normal|strong`: no balancing, global class balancing, or per-position class balancing.
- `--balance-target-classes`: also balance target classes when supported by the data loader.
- `--raw-sum-mod-10`: train raw-sum probes on `raw_sum % 10`.
- `--probe-mode dual_probe|mlp_direct`: use raw+carry probes or a direct digit MLP probe.

## Paper figure entry points

```bash
python figures/vertical_flow_umap.py
python figures/vertical_flow_pca.py
python figures/error_decomposition_umap.py
python figures/cluster_center_distance.py
```
