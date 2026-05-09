# Phase 2A locked hyperparameters

## qmix

**Locked config:** `qmix__embed_dim=16__mixer_lr=0.0001__mixer_init=orthogonal`

Mean over seeds:
- peak success: 0.517 ± 0.104
- final-window: 0.283 ± 0.132

Top 5 configurations by final-window success rate:

| config | final (mean) | final (std) | peak (mean) |
|---|---|---|---|
| `qmix__embed_dim=16__mixer_lr=0.0001__mixer_init=orthogonal` | 0.283 | 0.132 | 0.517 |
| `qmix__embed_dim=8__mixer_lr=0.0001__mixer_init=orthogonal` | 0.239 | 0.117 | 0.533 |
| `qmix__embed_dim=8__mixer_lr=0.0001__mixer_init=default` | 0.150 | 0.132 | 0.300 |
| `qmix__embed_dim=8__mixer_lr=0.0005__mixer_init=orthogonal` | 0.100 | 0.075 | 0.533 |
| `qmix__embed_dim=16__mixer_lr=0.0001__mixer_init=default` | 0.042 | 0.052 | 0.117 |


## gnn_qmix

**Locked config:** `gnn_qmix__embed_dim=8__mixer_lr=0.0001__mixer_init=orthogonal`

Mean over seeds:
- peak success: 0.667 ± 0.076
- final-window: 0.339 ± 0.079

Top 5 configurations by final-window success rate:

| config | final (mean) | final (std) | peak (mean) |
|---|---|---|---|
| `gnn_qmix__embed_dim=8__mixer_lr=0.0001__mixer_init=orthogonal` | 0.339 | 0.079 | 0.667 |
| `gnn_qmix__embed_dim=16__mixer_lr=0.0001__mixer_init=orthogonal` | 0.233 | 0.017 | 0.600 |
| `gnn_qmix__embed_dim=8__mixer_lr=0.0005__mixer_init=default` | 0.075 | 0.090 | 0.217 |
| `gnn_qmix__embed_dim=8__mixer_lr=0.0005__mixer_init=orthogonal` | 0.067 | 0.095 | 0.550 |
| `gnn_qmix__embed_dim=16__mixer_lr=0.0005__mixer_init=orthogonal` | 0.050 | 0.043 | 0.433 |


## iql, vdn

Not tuned in Phase 2A (Phase 1 confirmed defaults work). Lock to:

- `lr=5e-4`, `hidden=64` (PyTorch defaults), 30k steps in Phase 2C

