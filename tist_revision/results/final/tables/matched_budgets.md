### Matched budgets, L1 masked

**Clustering ($K{=}20, m{=}5$ / $K{=}50, m{=}5$ / $K{=}50, m{=}$all)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | 0.73<br><sub>[0.03]</sub> | 0.46<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.00]</sub> | 0.57<br><sub>[0.02]</sub> | 0.82<br><sub>[0.01]</sub> | 0.57<br><sub>[0.02]</sub> |
| k-means | TF+ToD | *0.77*<br><sub>[0.01]</sub> | 0.47<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.58<br><sub>[0.01]</sub> | 0.82<br><sub>[0.00]</sub> | 0.58<br><sub>[0.01]</sub> |
| k-means | Sensor+ToD hist. | *0.76*<br><sub>[0.00]</sub> | 0.44<br><sub>[0.02]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.55<br><sub>[0.01]</sub> | 0.82<br><sub>[0.00]</sub> | 0.60<br><sub>[0.01]</sub> |
| k-means | BERT+ToD | *0.76* | 0.44 | **0.82** | 0.59 | 0.82 | 0.60 |
| DEC | TF+ToD | **0.78**<br><sub>[0.01]</sub> | 0.47<br><sub>[0.01]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.02]</sub> | 0.80<br><sub>[0.00]</sub> | 0.53<br><sub>[0.01]</sub> |

**Linear probe on the same features (random activity labels)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear probe | TF+ToD | *0.76*<br><sub>[0.02]</sub> | *0.51*<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.03]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.66*<br><sub>[0.01]</sub> |
| Linear probe | Sensor+ToD hist. | **0.78**<br><sub>[0.02]</sub> | *0.51*<br><sub>[0.03]</sub> | **0.82**<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.03]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.67**<br><sub>[0.00]</sub> |
| Linear probe | BERT+ToD | 0.75 | **0.53** | *0.80* | **0.62** | *0.84* | *0.65* |

**Supervised baselines (random activity labels)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepCASAS | end-to-end | 0.61<br><sub>[0.02]</sub> | 0.31<br><sub>[0.03]</sub> | 0.66<br><sub>[0.01]</sub> | 0.40<br><sub>[0.02]</sub> | 0.73<br><sub>[0.00]</sub> | 0.57<br><sub>[0.00]</sub> |
| TDOST | end-to-end | 0.59<br><sub>[0.03]</sub> | 0.27<br><sub>[0.03]</sub> | 0.63<br><sub>[0.02]</sub> | 0.33<br><sub>[0.04]</sub> | *0.84*<br><sub>[0.00]</sub> | 0.62<br><sub>[0.01]</sub> |


### Matched budgets, L2 all

**Clustering ($K{=}20, m{=}5$ / $K{=}50, m{=}5$ / $K{=}50, m{=}$all)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | 0.78<br><sub>[0.02]</sub> | 0.37<br><sub>[0.02]</sub> | *0.83*<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.02]</sub> | 0.84<br><sub>[0.01]</sub> | 0.43<br><sub>[0.01]</sub> |
| k-means | TF+ToD | *0.80*<br><sub>[0.01]</sub> | 0.36<br><sub>[0.02]</sub> | *0.84*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.01]</sub> | *0.85*<br><sub>[0.00]</sub> | 0.46<br><sub>[0.00]</sub> |
| k-means | Sensor+ToD hist. | *0.79*<br><sub>[0.00]</sub> | 0.36<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.00]</sub> | *0.45*<br><sub>[0.01]</sub> | 0.84<br><sub>[0.00]</sub> | 0.47<br><sub>[0.01]</sub> |
| k-means | BERT+ToD | *0.79* | 0.32 | **0.84** | 0.43 | 0.84 | 0.43 |
| DEC | TF+ToD | *0.80*<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.02]</sub> | 0.83<br><sub>[0.00]</sub> | 0.43<br><sub>[0.01]</sub> |

**Linear probe on the same features (random activity labels)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| Linear probe | TF+ToD | *0.80*<br><sub>[0.01]</sub> | *0.40*<br><sub>[0.03]</sub> | *0.83*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.02]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.01]</sub> |
| Linear probe | Sensor+ToD hist. | **0.82**<br><sub>[0.01]</sub> | **0.41**<br><sub>[0.03]</sub> | **0.84**<br><sub>[0.01]</sub> | **0.47**<br><sub>[0.02]</sub> | **0.88**<br><sub>[0.00]</sub> | **0.58**<br><sub>[0.01]</sub> |
| Linear probe | BERT+ToD | *0.79* | *0.38* | *0.84* | **0.47** | *0.86* | 0.50 |

**Supervised baselines (random activity labels)**

| Method | Representation | ≈100 labels W | ≈100 labels M | ≈250 labels W | ≈250 labels M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepCASAS | end-to-end | 0.75<br><sub>[0.02]</sub> | 0.29<br><sub>[0.04]</sub> | 0.78<br><sub>[0.01]</sub> | 0.34<br><sub>[0.03]</sub> | 0.85<br><sub>[0.00]</sub> | 0.52<br><sub>[0.02]</sub> |
| TDOST | end-to-end | 0.74<br><sub>[0.02]</sub> | 0.25<br><sub>[0.04]</sub> | 0.77<br><sub>[0.01]</sub> | 0.30<br><sub>[0.03]</sub> | 0.84<br><sub>[0.04]</sub> | 0.47<br><sub>[0.05]</sub> |

