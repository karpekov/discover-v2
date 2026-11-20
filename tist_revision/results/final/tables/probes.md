### Probes, L1 masked (primary)

**budget = 100**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.73*<br><sub>[0.02]</sub> | 0.42<br><sub>[0.04]</sub> | *0.83*<br><sub>[0.03]</sub> | *0.52*<br><sub>[0.06]</sub> | 0.34<br><sub>[0.02]</sub> | 0.27<br><sub>[0.04]</sub> | *0.78*<br><sub>[0.02]</sub> | 0.56<br><sub>[0.05]</sub> | 0.67<br><sub>[0.02]</sub> | 0.44<br><sub>[0.03]</sub> |
| Linear probe | Sensor+ToD hist. | *0.73*<br><sub>[0.04]</sub> | 0.43<br><sub>[0.04]</sub> | *0.83*<br><sub>[0.03]</sub> | 0.50<br><sub>[0.07]</sub> | **0.77**<br><sub>[0.03]</sub> | **0.54**<br><sub>[0.05]</sub> | *0.79*<br><sub>[0.02]</sub> | 0.58<br><sub>[0.06]</sub> | **0.78**<br><sub>[0.02]</sub> | *0.51*<br><sub>[0.03]</sub> |
| Linear probe | TF | *0.74*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.02]</sub> | **0.53**<br><sub>[0.07]</sub> | 0.49<br><sub>[0.03]</sub> | 0.37<br><sub>[0.05]</sub> | **0.80**<br><sub>[0.02]</sub> | 0.59<br><sub>[0.03]</sub> | 0.72<br><sub>[0.01]</sub> | 0.49<br><sub>[0.03]</sub> |
| Linear probe | TF+ToD | *0.74*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.05]</sub> | **0.84**<br><sub>[0.02]</sub> | **0.53**<br><sub>[0.08]</sub> | 0.65<br><sub>[0.03]</sub> | 0.46<br><sub>[0.06]</sub> | **0.80**<br><sub>[0.02]</sub> | 0.59<br><sub>[0.03]</sub> | *0.76*<br><sub>[0.02]</sub> | *0.51*<br><sub>[0.04]</sub> |
| Linear probe | BERT | *0.75* | *0.50* | **0.84** | 0.49 | 0.33 | 0.34 | 0.74 | 0.48 | 0.66 | 0.45 |
| Linear probe | BERT+ToD | **0.76** | **0.52** | *0.83* | 0.47 | 0.61 | 0.51 | **0.79** | **0.63** | 0.75 | **0.53** |

**budget = 250**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.77*<br><sub>[0.01]</sub> | 0.53<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.60<br><sub>[0.02]</sub> | 0.36<br><sub>[0.02]</sub> | 0.32<br><sub>[0.05]</sub> | *0.81*<br><sub>[0.02]</sub> | 0.62<br><sub>[0.04]</sub> | 0.70<br><sub>[0.01]</sub> | 0.52<br><sub>[0.03]</sub> |
| Linear probe | Sensor+ToD hist. | **0.78**<br><sub>[0.01]</sub> | 0.56<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.59<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.61**<br><sub>[0.05]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.66*<br><sub>[0.05]</sub> | **0.82**<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.03]</sub> |
| Linear probe | TF | *0.77*<br><sub>[0.01]</sub> | 0.53<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.60<br><sub>[0.02]</sub> | 0.61<br><sub>[0.02]</sub> | 0.48<br><sub>[0.05]</sub> | **0.83**<br><sub>[0.01]</sub> | **0.68**<br><sub>[0.04]</sub> | 0.77<br><sub>[0.01]</sub> | 0.57<br><sub>[0.03]</sub> |
| Linear probe | TF+ToD | *0.77*<br><sub>[0.01]</sub> | 0.55<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.61<br><sub>[0.02]</sub> | 0.75<br><sub>[0.02]</sub> | 0.56<br><sub>[0.04]</sub> | **0.83**<br><sub>[0.01]</sub> | **0.68**<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.03]</sub> |
| Linear probe | BERT | *0.77* | *0.57* | **0.89** | **0.66** | 0.37 | 0.37 | 0.78 | 0.61 | 0.70 | 0.55 |
| Linear probe | BERT+ToD | **0.78** | **0.60** | **0.89** | **0.66** | 0.74 | *0.59* | *0.80* | 0.64 | *0.80* | **0.62** |

**all labels (max 20k)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.81*<br><sub>[0.00]</sub> | 0.62<br><sub>[0.00]</sub> | *0.89*<br><sub>[0.00]</sub> | *0.63*<br><sub>[0.00]</sub> | 0.38<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> | 0.83<br><sub>[0.00]</sub> | 0.64<br><sub>[0.00]</sub> | 0.73<br><sub>[0.00]</sub> | 0.57<br><sub>[0.00]</sub> |
| Linear probe | Sensor+ToD hist. | **0.83**<br><sub>[0.00]</sub> | **0.66**<br><sub>[0.00]</sub> | **0.89**<br><sub>[0.00]</sub> | *0.65*<br><sub>[0.00]</sub> | *0.83*<br><sub>[0.00]</sub> | *0.65*<br><sub>[0.00]</sub> | **0.86**<br><sub>[0.00]</sub> | **0.71**<br><sub>[0.00]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.67**<br><sub>[0.00]</sub> |
| Linear probe | TF | *0.81*<br><sub>[0.00]</sub> | 0.60<br><sub>[0.01]</sub> | **0.89**<br><sub>[0.00]</sub> | **0.66**<br><sub>[0.01]</sub> | *0.81*<br><sub>[0.01]</sub> | *0.66*<br><sub>[0.02]</sub> | 0.83<br><sub>[0.03]</sub> | *0.71*<br><sub>[0.04]</sub> | *0.84*<br><sub>[0.01]</sub> | *0.66*<br><sub>[0.01]</sub> |
| Linear probe | TF+ToD | *0.81*<br><sub>[0.00]</sub> | 0.60<br><sub>[0.01]</sub> | **0.89**<br><sub>[0.00]</sub> | **0.66**<br><sub>[0.01]</sub> | **0.84**<br><sub>[0.00]</sub> | **0.67**<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.02]</sub> | **0.71**<br><sub>[0.03]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.66*<br><sub>[0.01]</sub> |
| Linear probe | BERT | 0.79 | 0.58 | *0.88* | *0.64* | 0.40 | 0.38 | 0.81 | 0.67 | 0.72 | 0.57 |
| Linear probe | BERT+ToD | *0.80* | 0.60 | *0.88* | *0.64* | *0.82* | *0.65* | *0.84* | **0.71** | *0.84* | *0.65* |


### Probes, L1 all classes

**budget = 100**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.73*<br><sub>[0.02]</sub> | 0.34<br><sub>[0.05]</sub> | *0.83*<br><sub>[0.03]</sub> | **0.39**<br><sub>[0.04]</sub> | 0.34<br><sub>[0.02]</sub> | 0.25<br><sub>[0.05]</sub> | *0.78*<br><sub>[0.02]</sub> | *0.32*<br><sub>[0.09]</sub> | 0.67<br><sub>[0.02]</sub> | 0.33<br><sub>[0.04]</sub> |
| Linear probe | Sensor+ToD hist. | *0.73*<br><sub>[0.04]</sub> | 0.34<br><sub>[0.04]</sub> | *0.83*<br><sub>[0.03]</sub> | *0.37*<br><sub>[0.04]</sub> | **0.77**<br><sub>[0.03]</sub> | *0.50*<br><sub>[0.06]</sub> | *0.79*<br><sub>[0.02]</sub> | *0.32*<br><sub>[0.06]</sub> | **0.78**<br><sub>[0.02]</sub> | *0.38*<br><sub>[0.04]</sub> |
| Linear probe | TF | *0.74*<br><sub>[0.01]</sub> | 0.35<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.02]</sub> | *0.38*<br><sub>[0.04]</sub> | 0.49<br><sub>[0.03]</sub> | 0.35<br><sub>[0.06]</sub> | **0.80**<br><sub>[0.02]</sub> | **0.35**<br><sub>[0.07]</sub> | 0.72<br><sub>[0.01]</sub> | 0.36<br><sub>[0.04]</sub> |
| Linear probe | TF+ToD | *0.74*<br><sub>[0.01]</sub> | 0.35<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.02]</sub> | *0.38*<br><sub>[0.05]</sub> | 0.65<br><sub>[0.03]</sub> | 0.43<br><sub>[0.07]</sub> | **0.80**<br><sub>[0.02]</sub> | **0.35**<br><sub>[0.07]</sub> | *0.76*<br><sub>[0.02]</sub> | *0.37*<br><sub>[0.04]</sub> |
| Linear probe | BERT | *0.75* | *0.41* | **0.84** | 0.34 | 0.33 | 0.35 | 0.74 | 0.24 | 0.66 | 0.34 |
| Linear probe | BERT+ToD | **0.76** | **0.42** | *0.83* | 0.33 | 0.61 | **0.51** | **0.79** | 0.30 | 0.75 | **0.39** |

**budget = 250**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.77*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.04]</sub> | 0.36<br><sub>[0.02]</sub> | 0.30<br><sub>[0.05]</sub> | *0.81*<br><sub>[0.02]</sub> | *0.40*<br><sub>[0.05]</sub> | 0.70<br><sub>[0.01]</sub> | 0.40<br><sub>[0.03]</sub> |
| Linear probe | Sensor+ToD hist. | **0.78**<br><sub>[0.01]</sub> | 0.45<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.03]</sub> | **0.81**<br><sub>[0.02]</sub> | 0.56<br><sub>[0.05]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.05]</sub> | **0.82**<br><sub>[0.01]</sub> | **0.47**<br><sub>[0.03]</sub> |
| Linear probe | TF | *0.77*<br><sub>[0.01]</sub> | 0.44<br><sub>[0.05]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.03]</sub> | 0.61<br><sub>[0.02]</sub> | 0.44<br><sub>[0.05]</sub> | **0.83**<br><sub>[0.01]</sub> | **0.43**<br><sub>[0.06]</sub> | 0.77<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.03]</sub> |
| Linear probe | TF+ToD | *0.77*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.02]</sub> | 0.75<br><sub>[0.02]</sub> | 0.52<br><sub>[0.04]</sub> | **0.83**<br><sub>[0.01]</sub> | **0.42**<br><sub>[0.06]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.03]</sub> |
| Linear probe | BERT | *0.77* | *0.47* | **0.89** | **0.46** | 0.37 | 0.39 | 0.78 | 0.28 | 0.70 | 0.40 |
| Linear probe | BERT+ToD | **0.78** | **0.49** | **0.89** | **0.46** | 0.74 | **0.60** | *0.80* | 0.31 | *0.80* | **0.47** |

**all labels (max 20k)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.81*<br><sub>[0.00]</sub> | *0.52*<br><sub>[0.00]</sub> | *0.89*<br><sub>[0.00]</sub> | *0.55*<br><sub>[0.03]</sub> | 0.38<br><sub>[0.01]</sub> | 0.38<br><sub>[0.01]</sub> | 0.83<br><sub>[0.00]</sub> | *0.54*<br><sub>[0.00]</sub> | 0.73<br><sub>[0.00]</sub> | 0.50<br><sub>[0.01]</sub> |
| Linear probe | Sensor+ToD hist. | **0.83**<br><sub>[0.00]</sub> | *0.52*<br><sub>[0.03]</sub> | **0.89**<br><sub>[0.00]</sub> | **0.57**<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.00]</sub> | *0.64*<br><sub>[0.00]</sub> | **0.86**<br><sub>[0.00]</sub> | **0.56**<br><sub>[0.00]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.57**<br><sub>[0.01]</sub> |
| Linear probe | TF | *0.81*<br><sub>[0.00]</sub> | *0.49*<br><sub>[0.02]</sub> | **0.89**<br><sub>[0.00]</sub> | 0.44<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.01]</sub> | *0.63*<br><sub>[0.01]</sub> | 0.83<br><sub>[0.03]</sub> | *0.54*<br><sub>[0.04]</sub> | *0.84*<br><sub>[0.01]</sub> | 0.53<br><sub>[0.01]</sub> |
| Linear probe | TF+ToD | *0.81*<br><sub>[0.00]</sub> | 0.48<br><sub>[0.02]</sub> | **0.89**<br><sub>[0.00]</sub> | 0.48<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.00]</sub> | **0.65**<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.02]</sub> | *0.54*<br><sub>[0.04]</sub> | *0.85*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.02]</sub> |
| Linear probe | BERT | 0.79 | 0.49 | *0.88* | 0.43 | 0.40 | 0.38 | 0.81 | 0.38 | 0.72 | 0.42 |
| Linear probe | BERT+ToD | *0.80* | **0.52** | *0.88* | 0.46 | *0.82* | *0.63* | *0.84* | 0.46 | *0.84* | 0.52 |


### Probes, L2 all classes

**budget = 100**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.75*<br><sub>[0.01]</sub> | 0.33<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.03]</sub> | **0.45**<br><sub>[0.05]</sub> | 0.74<br><sub>[0.02]</sub> | 0.32<br><sub>[0.05]</sub> | **0.82**<br><sub>[0.03]</sub> | *0.42*<br><sub>[0.11]</sub> | *0.79*<br><sub>[0.01]</sub> | *0.38*<br><sub>[0.04]</sub> |
| Linear probe | Sensor+ToD hist. | **0.76**<br><sub>[0.01]</sub> | *0.36*<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.03]</sub> | *0.44*<br><sub>[0.05]</sub> | **0.85**<br><sub>[0.01]</sub> | 0.43<br><sub>[0.04]</sub> | **0.82**<br><sub>[0.02]</sub> | 0.41<br><sub>[0.07]</sub> | **0.82**<br><sub>[0.01]</sub> | **0.41**<br><sub>[0.03]</sub> |
| Linear probe | TF | *0.74*<br><sub>[0.02]</sub> | 0.33<br><sub>[0.05]</sub> | *0.85*<br><sub>[0.02]</sub> | *0.43*<br><sub>[0.04]</sub> | 0.79<br><sub>[0.02]</sub> | 0.38<br><sub>[0.05]</sub> | **0.82**<br><sub>[0.01]</sub> | **0.45**<br><sub>[0.07]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.40*<br><sub>[0.03]</sub> |
| Linear probe | TF+ToD | *0.74*<br><sub>[0.02]</sub> | 0.34<br><sub>[0.05]</sub> | *0.84*<br><sub>[0.02]</sub> | *0.43*<br><sub>[0.04]</sub> | *0.82*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.05]</sub> | **0.82**<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.06]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.40*<br><sub>[0.03]</sub> |
| Linear probe | BERT | *0.74* | *0.35* | **0.85** | 0.38 | 0.76 | 0.41 | 0.76 | 0.32 | 0.78 | 0.36 |
| Linear probe | BERT+ToD | *0.75* | **0.37** | *0.84* | 0.37 | 0.81 | **0.47** | 0.77 | 0.31 | *0.79* | *0.38* |

**budget = 250**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.78*<br><sub>[0.01]</sub> | 0.42<br><sub>[0.03]</sub> | *0.88*<br><sub>[0.01]</sub> | **0.52**<br><sub>[0.05]</sub> | 0.77<br><sub>[0.01]</sub> | 0.35<br><sub>[0.05]</sub> | *0.82*<br><sub>[0.02]</sub> | 0.44<br><sub>[0.07]</sub> | 0.81<br><sub>[0.00]</sub> | 0.43<br><sub>[0.03]</sub> |
| Linear probe | Sensor+ToD hist. | **0.79**<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | *0.51*<br><sub>[0.03]</sub> | **0.87**<br><sub>[0.01]</sub> | 0.48<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.01]</sub> | 0.45<br><sub>[0.05]</sub> | **0.84**<br><sub>[0.01]</sub> | **0.47**<br><sub>[0.02]</sub> |
| Linear probe | TF | *0.77*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.03]</sub> | 0.82<br><sub>[0.02]</sub> | 0.41<br><sub>[0.04]</sub> | *0.82*<br><sub>[0.02]</sub> | **0.48**<br><sub>[0.03]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.02]</sub> |
| Linear probe | TF+ToD | *0.78*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.02]</sub> | *0.85*<br><sub>[0.02]</sub> | 0.44<br><sub>[0.04]</sub> | *0.82*<br><sub>[0.02]</sub> | **0.49**<br><sub>[0.03]</sub> | *0.83*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.02]</sub> |
| Linear probe | BERT | *0.77* | *0.43* | **0.89** | 0.48 | 0.79 | 0.44 | 0.81 | 0.37 | 0.81 | 0.43 |
| Linear probe | BERT+ToD | *0.79* | **0.46** | **0.89** | 0.49 | *0.86* | **0.56** | *0.82* | 0.38 | *0.84* | **0.47** |

**all labels (max 20k)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.81*<br><sub>[0.00]</sub> | 0.49<br><sub>[0.00]</sub> | **0.90**<br><sub>[0.00]</sub> | *0.62*<br><sub>[0.04]</sub> | 0.81<br><sub>[0.00]</sub> | 0.43<br><sub>[0.02]</sub> | 0.84<br><sub>[0.00]</sub> | **0.58**<br><sub>[0.00]</sub> | 0.84<br><sub>[0.00]</sub> | 0.53<br><sub>[0.01]</sub> |
| Linear probe | Sensor+ToD hist. | **0.83**<br><sub>[0.00]</sub> | **0.55**<br><sub>[0.01]</sub> | **0.90**<br><sub>[0.00]</sub> | **0.63**<br><sub>[0.02]</sub> | **0.90**<br><sub>[0.00]</sub> | *0.56*<br><sub>[0.02]</sub> | **0.87**<br><sub>[0.00]</sub> | *0.57*<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.00]</sub> | **0.58**<br><sub>[0.01]</sub> |
| Linear probe | TF | *0.82*<br><sub>[0.00]</sub> | 0.50<br><sub>[0.02]</sub> | **0.90**<br><sub>[0.00]</sub> | 0.48<br><sub>[0.02]</sub> | *0.89*<br><sub>[0.00]</sub> | *0.56*<br><sub>[0.02]</sub> | 0.79<br><sub>[0.03]</sub> | *0.57*<br><sub>[0.03]</sub> | *0.85*<br><sub>[0.01]</sub> | 0.53<br><sub>[0.01]</sub> |
| Linear probe | TF+ToD | *0.82*<br><sub>[0.00]</sub> | 0.51<br><sub>[0.02]</sub> | **0.90**<br><sub>[0.00]</sub> | 0.51<br><sub>[0.03]</sub> | **0.90**<br><sub>[0.00]</sub> | **0.57**<br><sub>[0.02]</sub> | 0.80<br><sub>[0.04]</sub> | *0.57*<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.01]</sub> |
| Linear probe | BERT | 0.80 | 0.46 | *0.89* | 0.45 | 0.79 | 0.40 | 0.84 | 0.47 | 0.83 | 0.45 |
| Linear probe | BERT+ToD | *0.81* | 0.51 | *0.89* | 0.46 | *0.87* | 0.51 | *0.85* | 0.53 | *0.86* | 0.50 |

