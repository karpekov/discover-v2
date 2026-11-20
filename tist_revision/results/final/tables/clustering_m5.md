### L1 masked (primary); $m=5$

**$K = 20$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.75*<br><sub>[0.02]</sub> | **0.52**<br><sub>[0.04]</sub> | 0.84<br><sub>[0.02]</sub> | 0.59<br><sub>[0.06]</sub> | 0.27<br><sub>[0.02]</sub> | 0.27<br><sub>[0.02]</sub> | 0.76<br><sub>[0.01]</sub> | 0.46<br><sub>[0.04]</sub> | 0.65<br><sub>[0.01]</sub> | 0.46<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.69<br><sub>[0.02]</sub> | 0.34<br><sub>[0.02]</sub> | 0.77<br><sub>[0.02]</sub> | 0.38<br><sub>[0.03]</sub> | *0.78*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.03]</sub> | *0.76*<br><sub>[0.00]</sub> | 0.44<br><sub>[0.02]</sub> |
| k-means | TF | *0.75*<br><sub>[0.02]</sub> | 0.49<br><sub>[0.05]</sub> | **0.88**<br><sub>[0.00]</sub> | **0.64**<br><sub>[0.01]</sub> | 0.30<br><sub>[0.05]</sub> | 0.26<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.02]</sub> | *0.56*<br><sub>[0.04]</sub> | 0.68<br><sub>[0.01]</sub> | *0.49*<br><sub>[0.02]</sub> |
| k-means | TF+ToD | 0.68<br><sub>[0.04]</sub> | 0.36<br><sub>[0.03]</sub> | 0.80<br><sub>[0.01]</sub> | 0.40<br><sub>[0.04]</sub> | *0.79*<br><sub>[0.02]</sub> | 0.52<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.01]</sub> | *0.58*<br><sub>[0.03]</sub> | *0.77*<br><sub>[0.01]</sub> | *0.47*<br><sub>[0.02]</sub> |
| k-means | BERT | **0.76** | *0.50* | **0.88** | *0.64* | 0.33 | 0.33 | 0.74 | 0.51 | 0.68 | **0.50** |
| k-means | BERT+ToD | 0.68 | 0.31 | 0.82 | 0.42 | *0.80* | *0.56* | 0.75 | 0.46 | *0.76* | 0.44 |
| SCAN | TF | *0.75*<br><sub>[0.02]</sub> | 0.49<br><sub>[0.05]</sub> | 0.84<br><sub>[0.03]</sub> | 0.60<br><sub>[0.02]</sub> | 0.31<br><sub>[0.02]</sub> | 0.28<br><sub>[0.01]</sub> | 0.73<br><sub>[0.02]</sub> | 0.42<br><sub>[0.06]</sub> | 0.66<br><sub>[0.01]</sub> | 0.45<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | 0.73<br><sub>[0.04]</sub> | 0.46<br><sub>[0.07]</sub> | 0.81<br><sub>[0.03]</sub> | 0.50<br><sub>[0.03]</sub> | 0.65<br><sub>[0.07]</sub> | 0.41<br><sub>[0.05]</sub> | 0.75<br><sub>[0.02]</sub> | 0.46<br><sub>[0.06]</sub> | 0.73<br><sub>[0.03]</sub> | 0.46<br><sub>[0.04]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.76**<br><sub>[0.02]</sub> | *0.50*<br><sub>[0.04]</sub> | 0.77<br><sub>[0.04]</sub> | 0.56<br><sub>[0.05]</sub> | 0.41<br><sub>[0.06]</sub> | 0.34<br><sub>[0.06]</sub> | 0.73<br><sub>[0.01]</sub> | 0.41<br><sub>[0.06]</sub> | 0.67<br><sub>[0.02]</sub> | 0.45<br><sub>[0.03]</sub> |
| SCAN | BERT | 0.65 | 0.40 | *0.86* | 0.60 | 0.31 | 0.23 | 0.74 | 0.51 | 0.64 | 0.43 |
| SCAN (unweighted ToD) | BERT+ToD | **0.76** | 0.46 | *0.86* | 0.61 | 0.33 | 0.32 | 0.72 | 0.43 | 0.67 | 0.46 |
| SCAN | Sensor+ToD hist. | 0.73<br><sub>[0.04]</sub> | 0.44<br><sub>[0.06]</sub> | 0.83<br><sub>[0.03]</sub> | 0.54<br><sub>[0.02]</sub> | 0.68<br><sub>[0.06]</sub> | 0.44<br><sub>[0.04]</sub> | *0.76*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.02]</sub> | *0.75*<br><sub>[0.02]</sub> | *0.49*<br><sub>[0.03]</sub> |
| DEC | TF | *0.74*<br><sub>[0.00]</sub> | 0.47<br><sub>[0.02]</sub> | **0.88**<br><sub>[0.00]</sub> | *0.62*<br><sub>[0.02]</sub> | 0.31<br><sub>[0.03]</sub> | 0.26<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.02]</sub> | **0.59**<br><sub>[0.04]</sub> | 0.68<br><sub>[0.01]</sub> | *0.48*<br><sub>[0.01]</sub> |
| DEC | TF+ToD | 0.71<br><sub>[0.02]</sub> | 0.37<br><sub>[0.01]</sub> | 0.81<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.02]</sub> | 0.51<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.01]</sub> | *0.57*<br><sub>[0.03]</sub> | **0.78**<br><sub>[0.01]</sub> | *0.47*<br><sub>[0.01]</sub> |
| DEC | BERT | *0.73* | 0.44 | *0.86* | 0.59 | 0.29 | 0.27 | 0.16 | 0.34 | 0.51 | 0.41 |
| DEC | BERT+ToD | 0.69 | 0.34 | 0.83 | 0.47 | **0.81** | **0.57** | 0.62 | 0.55 | 0.74 | *0.48* |

**$K = 50$ clusters (250 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.74<br><sub>[0.03]</sub> | 0.49<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.61*<br><sub>[0.01]</sub> | 0.31<br><sub>[0.03]</sub> | 0.30<br><sub>[0.03]</sub> | 0.79<br><sub>[0.01]</sub> | 0.53<br><sub>[0.02]</sub> | 0.67<br><sub>[0.01]</sub> | 0.48<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.75<br><sub>[0.02]</sub> | 0.47<br><sub>[0.05]</sub> | 0.85<br><sub>[0.01]</sub> | 0.58<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.02]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.57<br><sub>[0.05]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.55<br><sub>[0.01]</sub> |
| k-means | TF | 0.76<br><sub>[0.01]</sub> | 0.50<br><sub>[0.02]</sub> | *0.88*<br><sub>[0.01]</sub> | *0.62*<br><sub>[0.02]</sub> | 0.34<br><sub>[0.04]</sub> | 0.33<br><sub>[0.03]</sub> | **0.83**<br><sub>[0.02]</sub> | *0.63*<br><sub>[0.03]</sub> | 0.70<br><sub>[0.01]</sub> | 0.52<br><sub>[0.02]</sub> |
| k-means | TF+ToD | 0.77<br><sub>[0.01]</sub> | 0.52<br><sub>[0.01]</sub> | 0.85<br><sub>[0.01]</sub> | 0.57<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.59*<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.62*<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.01]</sub> | *0.58*<br><sub>[0.01]</sub> |
| k-means | BERT | 0.77 | *0.55* | **0.88** | **0.63** | 0.32 | 0.34 | 0.74 | 0.49 | 0.68 | 0.50 |
| k-means | BERT+ToD | 0.77 | 0.53 | *0.86* | 0.59 | **0.83** | *0.60* | *0.81* | *0.64* | **0.82** | **0.59** |
| SCAN | TF | 0.77<br><sub>[0.02]</sub> | *0.56*<br><sub>[0.05]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.59<br><sub>[0.02]</sub> | 0.34<br><sub>[0.03]</sub> | 0.34<br><sub>[0.03]</sub> | *0.81*<br><sub>[0.01]</sub> | **0.64**<br><sub>[0.02]</sub> | 0.69<br><sub>[0.01]</sub> | 0.53<br><sub>[0.02]</sub> |
| SCAN | TF+ToD | *0.77*<br><sub>[0.02]</sub> | *0.53*<br><sub>[0.03]</sub> | 0.84<br><sub>[0.01]</sub> | 0.58<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | 0.54<br><sub>[0.02]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.61<br><sub>[0.05]</sub> | *0.80*<br><sub>[0.00]</sub> | *0.57*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.78*<br><sub>[0.01]</sub> | *0.54*<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.02]</sub> | 0.59<br><sub>[0.06]</sub> | 0.65<br><sub>[0.08]</sub> | 0.53<br><sub>[0.04]</sub> | *0.82*<br><sub>[0.01]</sub> | **0.64**<br><sub>[0.01]</sub> | 0.78<br><sub>[0.02]</sub> | *0.58*<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.76 | 0.49 | *0.87* | *0.63* | 0.36 | 0.34 | 0.70 | 0.32 | 0.67 | 0.44 |
| SCAN (unweighted ToD) | BERT+ToD | **0.80** | **0.56** | *0.86* | *0.61* | 0.36 | 0.38 | 0.76 | 0.47 | 0.69 | 0.50 |
| SCAN | Sensor+ToD hist. | 0.75<br><sub>[0.01]</sub> | 0.48<br><sub>[0.04]</sub> | 0.84<br><sub>[0.02]</sub> | 0.55<br><sub>[0.06]</sub> | 0.76<br><sub>[0.03]</sub> | 0.54<br><sub>[0.03]</sub> | 0.80<br><sub>[0.03]</sub> | 0.54<br><sub>[0.05]</sub> | 0.79<br><sub>[0.01]</sub> | 0.53<br><sub>[0.02]</sub> |
| DEC | TF | 0.76<br><sub>[0.01]</sub> | 0.50<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.58<br><sub>[0.02]</sub> | 0.32<br><sub>[0.02]</sub> | 0.28<br><sub>[0.03]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.61<br><sub>[0.04]</sub> | 0.69<br><sub>[0.00]</sub> | 0.49<br><sub>[0.02]</sub> |
| DEC | TF+ToD | 0.75<br><sub>[0.02]</sub> | 0.45<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.53<br><sub>[0.02]</sub> | 0.78<br><sub>[0.02]</sub> | 0.54<br><sub>[0.03]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.63*<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.02]</sub> |
| DEC | BERT | *0.78* | *0.55* | *0.87* | 0.60 | 0.33 | 0.30 | 0.14 | 0.32 | 0.53 | 0.44 |
| DEC | BERT+ToD | *0.78* | 0.51 | 0.85 | 0.56 | **0.83** | **0.62** | 0.63 | 0.57 | 0.77 | *0.57* |

**$K = 100$ clusters (500 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.78**<br><sub>[0.01]</sub> | **0.57**<br><sub>[0.02]</sub> | **0.87**<br><sub>[0.01]</sub> | **0.62**<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.61**<br><sub>[0.04]</sub> | **0.82**<br><sub>[0.01]</sub> | **0.61**<br><sub>[0.04]</sub> | **0.82**<br><sub>[0.00]</sub> | **0.60**<br><sub>[0.01]</sub> |


### L1 all classes; $m=5$

**$K = 20$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.75*<br><sub>[0.02]</sub> | **0.38**<br><sub>[0.03]</sub> | 0.84<br><sub>[0.02]</sub> | 0.42<br><sub>[0.04]</sub> | 0.27<br><sub>[0.02]</sub> | 0.24<br><sub>[0.02]</sub> | 0.76<br><sub>[0.01]</sub> | 0.28<br><sub>[0.04]</sub> | 0.65<br><sub>[0.01]</sub> | *0.33*<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.69<br><sub>[0.02]</sub> | 0.25<br><sub>[0.02]</sub> | 0.77<br><sub>[0.02]</sub> | 0.26<br><sub>[0.02]</sub> | *0.78*<br><sub>[0.01]</sub> | 0.47<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.32<br><sub>[0.04]</sub> | *0.76*<br><sub>[0.00]</sub> | *0.33*<br><sub>[0.02]</sub> |
| k-means | TF | *0.75*<br><sub>[0.02]</sub> | *0.36*<br><sub>[0.04]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.45<br><sub>[0.01]</sub> | 0.30<br><sub>[0.05]</sub> | 0.24<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.02]</sub> | *0.35*<br><sub>[0.03]</sub> | 0.68<br><sub>[0.01]</sub> | *0.35*<br><sub>[0.01]</sub> |
| k-means | TF+ToD | 0.68<br><sub>[0.04]</sub> | 0.27<br><sub>[0.03]</sub> | 0.80<br><sub>[0.01]</sub> | 0.28<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.02]</sub> | 0.48<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.02]</sub> | *0.77*<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.02]</sub> |
| k-means | BERT | **0.76** | *0.37* | **0.88** | **0.49** | 0.33 | 0.33 | 0.74 | 0.23 | 0.68 | **0.35** |
| k-means | BERT+ToD | 0.68 | 0.23 | 0.82 | 0.30 | *0.80* | *0.51* | 0.75 | 0.25 | *0.76* | 0.32 |
| SCAN | TF | *0.75*<br><sub>[0.02]</sub> | *0.36*<br><sub>[0.04]</sub> | 0.84<br><sub>[0.03]</sub> | 0.42<br><sub>[0.02]</sub> | 0.31<br><sub>[0.02]</sub> | 0.25<br><sub>[0.00]</sub> | 0.73<br><sub>[0.02]</sub> | 0.22<br><sub>[0.02]</sub> | 0.66<br><sub>[0.01]</sub> | 0.31<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | 0.73<br><sub>[0.04]</sub> | 0.34<br><sub>[0.05]</sub> | 0.81<br><sub>[0.03]</sub> | 0.35<br><sub>[0.02]</sub> | 0.65<br><sub>[0.07]</sub> | 0.38<br><sub>[0.05]</sub> | 0.75<br><sub>[0.02]</sub> | 0.25<br><sub>[0.02]</sub> | 0.73<br><sub>[0.03]</sub> | *0.33*<br><sub>[0.03]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.76**<br><sub>[0.02]</sub> | *0.36*<br><sub>[0.03]</sub> | 0.77<br><sub>[0.04]</sub> | 0.41<br><sub>[0.02]</sub> | 0.41<br><sub>[0.06]</sub> | 0.31<br><sub>[0.05]</sub> | 0.73<br><sub>[0.01]</sub> | 0.22<br><sub>[0.02]</sub> | 0.67<br><sub>[0.02]</sub> | 0.32<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.65 | 0.30 | *0.86* | 0.42 | 0.31 | 0.21 | 0.74 | 0.24 | 0.64 | 0.29 |
| SCAN (unweighted ToD) | BERT+ToD | **0.76** | *0.36* | *0.86* | 0.43 | 0.33 | 0.29 | 0.72 | 0.17 | 0.67 | 0.31 |
| SCAN | Sensor+ToD hist. | 0.73<br><sub>[0.04]</sub> | 0.32<br><sub>[0.04]</sub> | 0.83<br><sub>[0.03]</sub> | 0.38<br><sub>[0.02]</sub> | 0.68<br><sub>[0.06]</sub> | 0.40<br><sub>[0.04]</sub> | *0.76*<br><sub>[0.01]</sub> | 0.27<br><sub>[0.02]</sub> | *0.75*<br><sub>[0.02]</sub> | *0.35*<br><sub>[0.02]</sub> |
| DEC | TF | *0.74*<br><sub>[0.00]</sub> | 0.34<br><sub>[0.02]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.43<br><sub>[0.01]</sub> | 0.31<br><sub>[0.03]</sub> | 0.23<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.02]</sub> | **0.36**<br><sub>[0.04]</sub> | 0.68<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.01]</sub> |
| DEC | TF+ToD | 0.71<br><sub>[0.02]</sub> | 0.27<br><sub>[0.01]</sub> | 0.81<br><sub>[0.01]</sub> | 0.29<br><sub>[0.01]</sub> | *0.79*<br><sub>[0.02]</sub> | 0.47<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.01]</sub> | *0.33*<br><sub>[0.04]</sub> | **0.78**<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.01]</sub> |
| DEC | BERT | *0.73* | 0.33 | *0.86* | 0.42 | 0.29 | 0.24 | 0.16 | 0.17 | 0.51 | 0.29 |
| DEC | BERT+ToD | 0.69 | 0.25 | 0.83 | 0.33 | **0.81** | **0.52** | 0.62 | 0.29 | 0.74 | *0.35* |

**$K = 50$ clusters (250 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.74<br><sub>[0.03]</sub> | 0.36<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.44<br><sub>[0.03]</sub> | 0.31<br><sub>[0.03]</sub> | 0.28<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.03]</sub> | 0.67<br><sub>[0.01]</sub> | 0.38<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.75<br><sub>[0.02]</sub> | 0.35<br><sub>[0.03]</sub> | 0.85<br><sub>[0.01]</sub> | 0.42<br><sub>[0.03]</sub> | 0.79<br><sub>[0.01]</sub> | *0.54*<br><sub>[0.02]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.02]</sub> | *0.80*<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.01]</sub> |
| k-means | TF | 0.76<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> | *0.88*<br><sub>[0.01]</sub> | **0.51**<br><sub>[0.02]</sub> | 0.34<br><sub>[0.04]</sub> | 0.30<br><sub>[0.02]</sub> | **0.83**<br><sub>[0.02]</sub> | **0.46**<br><sub>[0.07]</sub> | 0.70<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> |
| k-means | TF+ToD | 0.77<br><sub>[0.01]</sub> | 0.38<br><sub>[0.01]</sub> | 0.85<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.54*<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | 0.42<br><sub>[0.05]</sub> | *0.81*<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.01]</sub> |
| k-means | BERT | 0.77 | **0.46** | **0.88** | 0.47 | 0.32 | 0.31 | 0.74 | 0.25 | 0.68 | 0.37 |
| k-means | BERT+ToD | 0.77 | 0.39 | *0.86* | 0.41 | **0.83** | **0.57** | *0.81* | 0.35 | **0.82** | *0.43* |
| SCAN | TF | 0.77<br><sub>[0.02]</sub> | 0.41<br><sub>[0.05]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.44<br><sub>[0.03]</sub> | 0.34<br><sub>[0.03]</sub> | 0.31<br><sub>[0.03]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.39<br><sub>[0.03]</sub> | 0.69<br><sub>[0.01]</sub> | 0.39<br><sub>[0.02]</sub> |
| SCAN | TF+ToD | *0.77*<br><sub>[0.02]</sub> | 0.40<br><sub>[0.03]</sub> | 0.84<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | 0.49<br><sub>[0.01]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.37<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.00]</sub> | *0.42*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.78*<br><sub>[0.01]</sub> | 0.40<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.02]</sub> | 0.44<br><sub>[0.02]</sub> | 0.65<br><sub>[0.08]</sub> | 0.48<br><sub>[0.04]</sub> | *0.82*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | 0.78<br><sub>[0.02]</sub> | **0.44**<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.76 | 0.40 | *0.87* | 0.45 | 0.36 | 0.31 | 0.70 | 0.19 | 0.67 | 0.34 |
| SCAN (unweighted ToD) | BERT+ToD | **0.80** | **0.46** | *0.86* | 0.45 | 0.36 | 0.34 | 0.76 | 0.29 | 0.69 | 0.38 |
| SCAN | Sensor+ToD hist. | 0.75<br><sub>[0.01]</sub> | 0.35<br><sub>[0.03]</sub> | 0.84<br><sub>[0.02]</sub> | 0.39<br><sub>[0.04]</sub> | 0.76<br><sub>[0.03]</sub> | 0.49<br><sub>[0.03]</sub> | 0.80<br><sub>[0.03]</sub> | 0.37<br><sub>[0.05]</sub> | 0.79<br><sub>[0.01]</sub> | 0.40<br><sub>[0.01]</sub> |
| DEC | TF | 0.76<br><sub>[0.01]</sub> | 0.38<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.01]</sub> | 0.32<br><sub>[0.02]</sub> | 0.25<br><sub>[0.03]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.42<br><sub>[0.04]</sub> | 0.69<br><sub>[0.00]</sub> | 0.37<br><sub>[0.01]</sub> |
| DEC | TF+ToD | 0.75<br><sub>[0.02]</sub> | 0.33<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.38<br><sub>[0.00]</sub> | 0.78<br><sub>[0.02]</sub> | 0.49<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.03]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.01]</sub> |
| DEC | BERT | *0.78* | *0.44* | *0.87* | 0.42 | 0.33 | 0.27 | 0.14 | 0.19 | 0.53 | 0.33 |
| DEC | BERT+ToD | *0.78* | 0.38 | 0.85 | 0.39 | **0.83** | **0.56** | 0.63 | 0.36 | 0.77 | *0.42* |

**$K = 100$ clusters (500 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.78**<br><sub>[0.01]</sub> | **0.42**<br><sub>[0.02]</sub> | **0.87**<br><sub>[0.01]</sub> | **0.47**<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.57**<br><sub>[0.04]</sub> | **0.82**<br><sub>[0.01]</sub> | **0.45**<br><sub>[0.03]</sub> | **0.82**<br><sub>[0.00]</sub> | **0.48**<br><sub>[0.01]</sub> |


### L2 all classes; $m=5$

**$K = 20$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.77**<br><sub>[0.02]</sub> | **0.43**<br><sub>[0.02]</sub> | 0.84<br><sub>[0.02]</sub> | 0.46<br><sub>[0.05]</sub> | 0.72<br><sub>[0.02]</sub> | 0.29<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | 0.36<br><sub>[0.03]</sub> | *0.78*<br><sub>[0.01]</sub> | 0.38<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.71<br><sub>[0.02]</sub> | 0.28<br><sub>[0.02]</sub> | 0.77<br><sub>[0.02]</sub> | 0.31<br><sub>[0.02]</sub> | *0.84*<br><sub>[0.00]</sub> | *0.40*<br><sub>[0.01]</sub> | **0.84**<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.00]</sub> | 0.36<br><sub>[0.01]</sub> |
| k-means | TF | *0.76*<br><sub>[0.03]</sub> | 0.37<br><sub>[0.03]</sub> | *0.89*<br><sub>[0.00]</sub> | 0.51<br><sub>[0.00]</sub> | 0.74<br><sub>[0.03]</sub> | 0.29<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.02]</sub> | *0.42*<br><sub>[0.07]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.40*<br><sub>[0.02]</sub> |
| k-means | TF+ToD | 0.70<br><sub>[0.03]</sub> | 0.30<br><sub>[0.01]</sub> | 0.81<br><sub>[0.01]</sub> | 0.33<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.01]</sub> | 0.39<br><sub>[0.01]</sub> | *0.83*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.07]</sub> | *0.80*<br><sub>[0.01]</sub> | 0.36<br><sub>[0.02]</sub> |
| k-means | BERT | *0.76* | *0.40* | **0.89** | **0.55** | 0.77 | 0.39 | 0.78 | 0.34 | **0.80** | **0.42** |
| k-means | BERT+ToD | 0.70 | 0.26 | 0.83 | 0.35 | *0.86* | *0.40* | 0.78 | 0.28 | *0.79* | 0.32 |
| SCAN | TF | *0.76*<br><sub>[0.01]</sub> | 0.39<br><sub>[0.02]</sub> | 0.85<br><sub>[0.04]</sub> | 0.47<br><sub>[0.03]</sub> | 0.77<br><sub>[0.01]</sub> | 0.34<br><sub>[0.00]</sub> | 0.76<br><sub>[0.02]</sub> | 0.29<br><sub>[0.03]</sub> | *0.78*<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | *0.75*<br><sub>[0.03]</sub> | 0.37<br><sub>[0.05]</sub> | 0.80<br><sub>[0.05]</sub> | 0.38<br><sub>[0.05]</sub> | 0.75<br><sub>[0.04]</sub> | 0.34<br><sub>[0.04]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.39<br><sub>[0.01]</sub> | *0.78*<br><sub>[0.02]</sub> | 0.37<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.77**<br><sub>[0.02]</sub> | 0.39<br><sub>[0.03]</sub> | 0.76<br><sub>[0.04]</sub> | 0.43<br><sub>[0.03]</sub> | 0.80<br><sub>[0.05]</sub> | 0.37<br><sub>[0.05]</sub> | 0.78<br><sub>[0.02]</sub> | 0.31<br><sub>[0.04]</sub> | *0.78*<br><sub>[0.02]</sub> | 0.38<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.69 | 0.37 | *0.86* | 0.47 | 0.77 | 0.31 | 0.77 | 0.33 | 0.77 | 0.37 |
| SCAN (unweighted ToD) | BERT+ToD | **0.77** | 0.38 | *0.87* | 0.48 | 0.78 | 0.36 | 0.72 | 0.23 | *0.78* | 0.36 |
| SCAN | Sensor+ToD hist. | *0.74*<br><sub>[0.03]</sub> | 0.36<br><sub>[0.05]</sub> | 0.84<br><sub>[0.03]</sub> | 0.41<br><sub>[0.02]</sub> | 0.80<br><sub>[0.04]</sub> | 0.38<br><sub>[0.03]</sub> | 0.80<br><sub>[0.02]</sub> | 0.37<br><sub>[0.06]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.38<br><sub>[0.03]</sub> |
| DEC | TF | *0.75*<br><sub>[0.00]</sub> | 0.34<br><sub>[0.02]</sub> | *0.88*<br><sub>[0.00]</sub> | 0.47<br><sub>[0.01]</sub> | 0.77<br><sub>[0.01]</sub> | 0.32<br><sub>[0.01]</sub> | 0.81<br><sub>[0.04]</sub> | *0.44*<br><sub>[0.04]</sub> | **0.80**<br><sub>[0.01]</sub> | 0.39<br><sub>[0.01]</sub> |
| DEC | TF+ToD | 0.71<br><sub>[0.01]</sub> | 0.30<br><sub>[0.02]</sub> | 0.82<br><sub>[0.01]</sub> | 0.36<br><sub>[0.01]</sub> | *0.84*<br><sub>[0.02]</sub> | 0.39<br><sub>[0.03]</sub> | **0.83**<br><sub>[0.01]</sub> | *0.42*<br><sub>[0.02]</sub> | **0.80**<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> |
| DEC | BERT | 0.73 | 0.31 | *0.87* | 0.45 | 0.75 | 0.29 | 0.13 | 0.24 | 0.62 | 0.32 |
| DEC | BERT+ToD | 0.68 | 0.26 | 0.83 | 0.37 | **0.87** | **0.42** | 0.59 | 0.26 | 0.74 | 0.33 |

**$K = 50$ clusters (250 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.75<br><sub>[0.02]</sub> | 0.36<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.50<br><sub>[0.04]</sub> | 0.75<br><sub>[0.01]</sub> | 0.32<br><sub>[0.03]</sub> | 0.82<br><sub>[0.02]</sub> | **0.55**<br><sub>[0.02]</sub> | 0.80<br><sub>[0.01]</sub> | 0.43<br><sub>[0.01]</sub> |
| k-means | Sensor+ToD hist. | 0.76<br><sub>[0.01]</sub> | 0.38<br><sub>[0.01]</sub> | 0.86<br><sub>[0.01]</sub> | 0.46<br><sub>[0.04]</sub> | 0.85<br><sub>[0.00]</sub> | 0.42<br><sub>[0.01]</sub> | *0.84*<br><sub>[0.00]</sub> | *0.53*<br><sub>[0.02]</sub> | *0.83*<br><sub>[0.00]</sub> | *0.45*<br><sub>[0.01]</sub> |
| k-means | TF | 0.77<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> | *0.89*<br><sub>[0.01]</sub> | **0.58**<br><sub>[0.02]</sub> | 0.78<br><sub>[0.01]</sub> | 0.35<br><sub>[0.01]</sub> | **0.86**<br><sub>[0.02]</sub> | **0.56**<br><sub>[0.07]</sub> | *0.82*<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.02]</sub> |
| k-means | TF+ToD | *0.78*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | 0.85<br><sub>[0.02]</sub> | 0.46<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.01]</sub> | *0.85*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.01]</sub> |
| k-means | BERT | *0.78* | 0.42 | **0.89** | 0.53 | 0.78 | 0.36 | 0.79 | 0.33 | 0.81 | 0.41 |
| k-means | BERT+ToD | *0.79* | 0.43 | *0.87* | 0.45 | *0.88* | **0.46** | *0.84* | 0.37 | **0.84** | 0.43 |
| SCAN | TF | *0.77*<br><sub>[0.02]</sub> | *0.44*<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.47<br><sub>[0.03]</sub> | 0.78<br><sub>[0.02]</sub> | 0.35<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.07]</sub> | 0.81<br><sub>[0.01]</sub> | *0.44*<br><sub>[0.02]</sub> |
| SCAN | TF+ToD | *0.78*<br><sub>[0.02]</sub> | *0.44*<br><sub>[0.03]</sub> | 0.85<br><sub>[0.02]</sub> | 0.45<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.02]</sub> | *0.45*<br><sub>[0.02]</sub> | *0.83*<br><sub>[0.02]</sub> | 0.47<br><sub>[0.04]</sub> | *0.83*<br><sub>[0.01]</sub> | *0.45*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.78*<br><sub>[0.02]</sub> | *0.44*<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.02]</sub> | 0.49<br><sub>[0.03]</sub> | 0.83<br><sub>[0.03]</sub> | 0.42<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.01]</sub> | 0.50<br><sub>[0.04]</sub> | *0.83*<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.01]</sub> |
| SCAN | BERT | 0.77 | 0.39 | *0.88* | 0.49 | 0.79 | 0.37 | 0.77 | 0.36 | 0.80 | 0.40 |
| SCAN (unweighted ToD) | BERT+ToD | **0.80** | **0.46** | 0.85 | 0.47 | 0.78 | 0.37 | 0.78 | 0.35 | 0.80 | 0.41 |
| SCAN | Sensor+ToD hist. | 0.76<br><sub>[0.02]</sub> | 0.38<br><sub>[0.04]</sub> | 0.84<br><sub>[0.02]</sub> | 0.42<br><sub>[0.02]</sub> | 0.82<br><sub>[0.04]</sub> | 0.39<br><sub>[0.03]</sub> | *0.84*<br><sub>[0.02]</sub> | 0.52<br><sub>[0.03]</sub> | *0.82*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.01]</sub> |
| DEC | TF | 0.77<br><sub>[0.01]</sub> | 0.37<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.01]</sub> | 0.77<br><sub>[0.01]</sub> | 0.33<br><sub>[0.02]</sub> | *0.84*<br><sub>[0.01]</sub> | *0.54*<br><sub>[0.03]</sub> | 0.81<br><sub>[0.00]</sub> | 0.42<br><sub>[0.01]</sub> |
| DEC | TF+ToD | 0.76<br><sub>[0.02]</sub> | 0.37<br><sub>[0.03]</sub> | 0.86<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | 0.84<br><sub>[0.02]</sub> | 0.42<br><sub>[0.02]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.53*<br><sub>[0.06]</sub> | *0.83*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.02]</sub> |
| DEC | BERT | *0.78* | 0.42 | *0.88* | 0.46 | 0.78 | 0.35 | 0.13 | 0.23 | 0.64 | 0.37 |
| DEC | BERT+ToD | *0.78* | 0.40 | 0.85 | 0.42 | **0.89** | **0.46** | 0.62 | 0.35 | 0.79 | 0.41 |

**$K = 100$ clusters (500 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.79**<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.01]</sub> | **0.87**<br><sub>[0.01]</sub> | **0.51**<br><sub>[0.02]</sub> | **0.87**<br><sub>[0.02]</sub> | **0.48**<br><sub>[0.02]</sub> | **0.85**<br><sub>[0.01]</sub> | **0.56**<br><sub>[0.04]</sub> | **0.84**<br><sub>[0.00]</sub> | **0.50**<br><sub>[0.00]</sub> |

