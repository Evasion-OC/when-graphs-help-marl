# H1 statistical tests

Per-scale pairwise comparisons on main-sweep results.
Holm correction across 6 pairs per (scale, metric).

## scale=small, metric=peak

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.680 | 0.470 | +0.210 | +2.51 | 0.0263 | 0.0901 | **yes** |
| iql vs qmix | 0.680 | 0.590 | +0.090 | +1.20 | 0.3351 | 0.3979 | no |
| iql vs gnn_qmix | 0.680 | 0.630 | +0.050 | +0.63 | 0.4892 | 0.5562 | no |
| vdn vs qmix | 0.470 | 0.590 | -0.120 | -2.15 | 0.0493 | 0.0901 | **yes** |
| vdn vs gnn_qmix | 0.470 | 0.630 | -0.160 | -2.57 | 0.0229 | 0.0901 | **yes** |
| qmix vs gnn_qmix | 0.590 | 0.630 | -0.040 | -0.80 | 0.4892 | 0.5562 | no |

## scale=small, metric=final

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.319 | 0.253 | +0.067 | +0.98 | 0.5039 | 0.6219 | no |
| iql vs qmix | 0.319 | 0.378 | -0.059 | -0.86 | 0.5039 | 0.6219 | no |
| iql vs gnn_qmix | 0.319 | 0.428 | -0.109 | -1.29 | 0.3030 | 0.3810 | no |
| vdn vs qmix | 0.253 | 0.378 | -0.126 | -2.40 | 0.0314 | 0.0952 | **yes** |
| vdn vs gnn_qmix | 0.253 | 0.428 | -0.176 | -2.45 | 0.0339 | 0.0952 | **yes** |
| qmix vs gnn_qmix | 0.378 | 0.428 | -0.050 | -0.70 | 0.5039 | 0.6219 | no |

## scale=medium, metric=peak

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.200 | 0.010 | +0.190 | +4.12 | 0.0074 | 0.0554 | **yes** |
| iql vs qmix | 0.200 | 0.040 | +0.160 | +3.47 | 0.0133 | 0.0554 | **yes** |
| iql vs gnn_qmix | 0.200 | 0.040 | +0.160 | +3.47 | 0.0133 | 0.0554 | **yes** |
| vdn vs qmix | 0.010 | 0.040 | -0.030 | -1.34 | 0.2001 | 0.2789 | no |
| vdn vs gnn_qmix | 0.010 | 0.040 | -0.030 | -1.34 | 0.2001 | 0.2789 | no |
| qmix vs gnn_qmix | 0.040 | 0.040 | +0.000 | +0.00 | 1.0000 | 1.0000 | no |

## scale=medium, metric=final

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.117 | 0.000 | +0.117 | +4.43 | 0.0102 | 0.0450 | **yes** |
| iql vs qmix | 0.117 | 0.007 | +0.110 | +4.05 | 0.0102 | 0.0545 | **yes** |
| iql vs gnn_qmix | 0.117 | 0.013 | +0.103 | +3.52 | 0.0095 | 0.0545 | **yes** |
| vdn vs qmix | 0.000 | 0.007 | -0.007 | -1.03 | 0.5334 | 0.5310 | no |
| vdn vs gnn_qmix | 0.000 | 0.013 | -0.013 | -1.03 | 0.5334 | 0.5310 | no |
| qmix vs gnn_qmix | 0.007 | 0.013 | -0.007 | -0.46 | 0.5334 | 0.7220 | no |

## scale=large, metric=peak

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.020 | 0.000 | +0.020 | +1.03 | 1.0000 | 1.0000 | no |
| iql vs qmix | 0.020 | 0.000 | +0.020 | +1.03 | 1.0000 | 1.0000 | no |
| iql vs gnn_qmix | 0.020 | 0.000 | +0.020 | +1.03 | 1.0000 | 1.0000 | no |
| vdn vs qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |
| vdn vs gnn_qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |
| qmix vs gnn_qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |

## scale=large, metric=final

| A vs B | mean_A | mean_B | Δ | d | Welch p (Holm) | MWU p (Holm) | sig |
|---|---|---|---|---|---|---|---|
| iql vs vdn | 0.010 | 0.000 | +0.010 | +0.63 | 1.0000 | 1.0000 | no |
| iql vs qmix | 0.010 | 0.000 | +0.010 | +0.63 | 1.0000 | 1.0000 | no |
| iql vs gnn_qmix | 0.010 | 0.000 | +0.010 | +0.63 | 1.0000 | 1.0000 | no |
| vdn vs qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |
| vdn vs gnn_qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |
| qmix vs gnn_qmix | 0.000 | 0.000 | +0.000 | +nan | 1.0000 | 1.0000 | no |

