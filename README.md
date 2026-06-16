# Temporal Weather GNN

A **Dynamic Spatio-Temporal Graph Neural Network** for multi-step weather forecasting, featuring:

- **Dynamic graph construction** – learned adjacency that evolves per time step
- **Temporal Attention Encoder** – Transformer across the time axis per node
- **Ripple Propagation Processor** – localized wavefront message-passing for efficiency
- **Static + Dynamic Graph Fusion** – gated, attention, or residual fusion modes
- **Autoregressive rollout** – GRU-based multi-step forecasting
- **Full metric suite** – RMSE, MAE, Bias, ACC, Skill Scores vs. persistence & climatology

---

### 1. Install dependencies

```bash
# Create environment
conda create -n weather_gnn python=3.10
conda activate weather_gnn

# PyTorch (CPU example; adjust for CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# PyG
pip install torch-geometric

# Everything else
pip install -r requirements.txt
```

### 2. Generate synthetic data

```bash
python data/prepare_data.py --source synthetic --n_stations 100
```

### 3. Train

```bash
# Full model (static + dynamic + fusion)
python train.py --config configs/default.yaml

# Ablation: static graph only
python train.py --config configs/default.yaml model.variant=static

# Ablation: dynamic graph only
python train.py --config configs/default.yaml model.variant=dynamic

# Custom hyperparameters
python train.py --config configs/default.yaml model.hidden_dim=256 training.epochs=100
```

### 4. Evaluate

```bash
# Evaluate with saved checkpoint
python evaluate.py --checkpoint checkpoints/best-epoch=10-val/loss=0.1234.ckpt

# Quick sanity check (random weights)
python evaluate.py --checkpoint none
```



### 5. Real data (NOAA ISD)

```bash
python data/prepare_data.py --source noaa --year 2022 --n_stations 50
python train.py --config configs/default.yaml data.dataset=noaa
```

### 6. Real data (ERA5)

```bash
# Requires ~/.cdsapirc with CDS credentials
# See: https://cds.climate.copernicus.eu/how-to-api
python data/prepare_data.py --source era5 --year 2022
python train.py --config configs/default.yaml data.dataset=era5
```


## Key Design Decisions

### Ripple Propagation
Traditional GNNs apply message-passing uniformly across all N nodes at every layer.
Ripple Propagation instead:
1. Scores node importance from the hidden state.
2. Selects epicentre nodes above a threshold.
3. Expands a localized "active front" via BFS for `radius` hops.
4. Runs a full GAT layer **only on the active front**.

On large graphs (N ~ 10k+), this reduces per-layer cost from O(N) to O(|front|),
while concentrating computation on nodes undergoing rapid change (storms, fronts).

### Static + Dynamic Fusion
- **Static graph** encodes geography: stable k-NN connectivity ensures global coverage.
- **Dynamic graph** captures evolving atmospheric correlations: wind patterns, pressure systems.
- **Gated fusion** (default) lets the model learn how much to trust each graph per time step.

### Autoregressive Rollout
Multi-step predictions are generated autoregressively:
- Each step re-builds the dynamic graph from the updated hidden state.
- A GRU cell updates the node hidden state using the previous prediction.
- A light re-fusion + ripple pass incorporates spatial context.

---

## References

- Neural-LAM: Oskarsson et al. (2023) — *Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks*
- GraphCast: Lam et al. (2023) — *Learning skillful medium-range global weather forecasting*
- T-RippleGNN: Ripple propagation for spatio-temporal traffic graphs
- TGN: Rossi et al. (2020) — *Temporal Graph Networks for Deep Learning on Dynamic Graphs*
- GAT: Veličković et al. (2018) — *Graph Attention Networks*
