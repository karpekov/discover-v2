### L1 masked (primary); $m=1$

**$K = 20$ clusters (20 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.66<br><sub>[0.01]</sub> | 0.43<br><sub>[0.01]</sub> | 0.83<br><sub>[0.03]</sub> | *0.61*<br><sub>[0.05]</sub> | 0.27<br><sub>[0.03]</sub> | 0.27<br><sub>[0.02]</sub> | 0.59<br><sub>[0.02]</sub> | *0.44*<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.56<br><sub>[0.11]</sub> | 0.32<br><sub>[0.06]</sub> | 0.78<br><sub>[0.02]</sub> | 0.40<br><sub>[0.04]</sub> | 0.75<br><sub>[0.00]</sub> | 0.46<br><sub>[0.02]</sub> | 0.69<br><sub>[0.03]</sub> | 0.39<br><sub>[0.02]</sub> |
| k-means | TF | *0.74*<br><sub>[0.03]</sub> | *0.46*<br><sub>[0.05]</sub> | *0.87*<br><sub>[0.01]</sub> | *0.61*<br><sub>[0.04]</sub> | 0.31<br><sub>[0.02]</sub> | 0.26<br><sub>[0.01]</sub> | 0.64<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.03]</sub> |
| k-means | TF+ToD | 0.67<br><sub>[0.05]</sub> | 0.35<br><sub>[0.04]</sub> | 0.78<br><sub>[0.02]</sub> | 0.39<br><sub>[0.03]</sub> | *0.78*<br><sub>[0.04]</sub> | 0.51<br><sub>[0.04]</sub> | *0.74*<br><sub>[0.03]</sub> | *0.42*<br><sub>[0.03]</sub> |
| k-means | BERT | 0.71 | 0.40 | **0.88** | **0.62** | 0.28 | 0.28 | 0.62 | *0.43* |
| k-means | BERT+ToD | 0.69 | 0.36 | 0.82 | 0.41 | *0.77* | **0.54** | **0.76** | *0.44* |
| SCAN | TF | *0.73*<br><sub>[0.02]</sub> | **0.46**<br><sub>[0.05]</sub> | 0.80<br><sub>[0.05]</sub> | 0.57<br><sub>[0.04]</sub> | 0.32<br><sub>[0.03]</sub> | 0.25<br><sub>[0.06]</sub> | 0.62<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | 0.69<br><sub>[0.03]</sub> | 0.41<br><sub>[0.05]</sub> | 0.79<br><sub>[0.05]</sub> | 0.43<br><sub>[0.06]</sub> | 0.60<br><sub>[0.07]</sub> | 0.38<br><sub>[0.06]</sub> | 0.69<br><sub>[0.05]</sub> | 0.41<br><sub>[0.05]</sub> |
| SCAN (unweighted ToD) | TF+ToD | 0.71<br><sub>[0.04]</sub> | **0.47**<br><sub>[0.05]</sub> | 0.78<br><sub>[0.04]</sub> | 0.54<br><sub>[0.06]</sub> | 0.38<br><sub>[0.03]</sub> | 0.31<br><sub>[0.05]</sub> | 0.62<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.03]</sub> |
| SCAN | BERT | 0.60 | 0.33 | *0.86* | *0.60* | 0.28 | 0.18 | 0.58 | 0.37 |
| SCAN (unweighted ToD) | BERT+ToD | **0.76** | *0.44* | 0.73 | 0.55 | 0.25 | 0.25 | 0.58 | *0.41* |
| SCAN | Sensor+ToD hist. | 0.63<br><sub>[0.03]</sub> | 0.38<br><sub>[0.08]</sub> | 0.80<br><sub>[0.05]</sub> | 0.50<br><sub>[0.01]</sub> | 0.65<br><sub>[0.08]</sub> | 0.41<br><sub>[0.07]</sub> | 0.69<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.04]</sub> |
| DEC | TF | 0.71<br><sub>[0.04]</sub> | 0.40<br><sub>[0.05]</sub> | 0.84<br><sub>[0.05]</sub> | 0.55<br><sub>[0.09]</sub> | 0.29<br><sub>[0.02]</sub> | 0.25<br><sub>[0.02]</sub> | 0.61<br><sub>[0.02]</sub> | 0.40<br><sub>[0.03]</sub> |
| DEC | TF+ToD | 0.68<br><sub>[0.04]</sub> | 0.37<br><sub>[0.04]</sub> | 0.79<br><sub>[0.03]</sub> | 0.39<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.03]</sub> | *0.52*<br><sub>[0.03]</sub> | **0.76**<br><sub>[0.02]</sub> | *0.42*<br><sub>[0.03]</sub> |
| DEC | BERT | 0.71 | 0.36 | **0.88** | **0.62** | 0.32 | 0.30 | 0.63 | *0.43* |
| DEC | BERT+ToD | 0.68 | 0.28 | 0.82 | 0.46 | *0.77* | *0.53* | *0.76* | *0.42* |

**$K = 50$ clusters (50 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.73*<br><sub>[0.02]</sub> | 0.46<br><sub>[0.04]</sub> | 0.83<br><sub>[0.03]</sub> | 0.58<br><sub>[0.03]</sub> | 0.28<br><sub>[0.02]</sub> | 0.28<br><sub>[0.01]</sub> | 0.62<br><sub>[0.02]</sub> | 0.44<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.70<br><sub>[0.03]</sub> | 0.42<br><sub>[0.03]</sub> | 0.79<br><sub>[0.03]</sub> | 0.53<br><sub>[0.03]</sub> | 0.73<br><sub>[0.03]</sub> | 0.52<br><sub>[0.03]</sub> | 0.74<br><sub>[0.01]</sub> | 0.49<br><sub>[0.02]</sub> |
| k-means | TF | *0.75*<br><sub>[0.02]</sub> | *0.51*<br><sub>[0.02]</sub> | *0.86*<br><sub>[0.01]</sub> | *0.62*<br><sub>[0.02]</sub> | 0.35<br><sub>[0.03]</sub> | 0.33<br><sub>[0.02]</sub> | 0.65<br><sub>[0.02]</sub> | 0.48<br><sub>[0.01]</sub> |
| k-means | TF+ToD | 0.73<br><sub>[0.02]</sub> | 0.48<br><sub>[0.02]</sub> | 0.85<br><sub>[0.01]</sub> | 0.56<br><sub>[0.04]</sub> | 0.78<br><sub>[0.02]</sub> | 0.55<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | *0.53*<br><sub>[0.01]</sub> |
| k-means | BERT | **0.76** | **0.53** | **0.88** | **0.63** | 0.27 | 0.28 | 0.64 | 0.48 |
| k-means | BERT+ToD | *0.75* | 0.49 | *0.86* | 0.58 | 0.80 | 0.57 | **0.80** | **0.55** |
| SCAN | TF | *0.74*<br><sub>[0.03]</sub> | 0.47<br><sub>[0.06]</sub> | 0.84<br><sub>[0.02]</sub> | 0.60<br><sub>[0.03]</sub> | 0.34<br><sub>[0.01]</sub> | 0.34<br><sub>[0.02]</sub> | 0.64<br><sub>[0.00]</sub> | 0.47<br><sub>[0.03]</sub> |
| SCAN | TF+ToD | *0.75*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.02]</sub> | 0.83<br><sub>[0.02]</sub> | 0.55<br><sub>[0.03]</sub> | 0.78<br><sub>[0.02]</sub> | 0.53<br><sub>[0.01]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.75*<br><sub>[0.02]</sub> | 0.49<br><sub>[0.02]</sub> | 0.84<br><sub>[0.03]</sub> | 0.57<br><sub>[0.06]</sub> | 0.59<br><sub>[0.06]</sub> | 0.47<br><sub>[0.02]</sub> | 0.72<br><sub>[0.03]</sub> | 0.51<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.70 | 0.36 | *0.87* | *0.61* | 0.32 | 0.33 | 0.63 | 0.43 |
| SCAN (unweighted ToD) | BERT+ToD | *0.75* | 0.45 | 0.80 | 0.60 | 0.38 | 0.36 | 0.64 | 0.47 |
| SCAN | Sensor+ToD hist. | 0.70<br><sub>[0.01]</sub> | 0.46<br><sub>[0.03]</sub> | 0.83<br><sub>[0.03]</sub> | 0.54<br><sub>[0.08]</sub> | 0.73<br><sub>[0.06]</sub> | 0.51<br><sub>[0.08]</sub> | 0.75<br><sub>[0.02]</sub> | 0.50<br><sub>[0.03]</sub> |
| DEC | TF | 0.70<br><sub>[0.07]</sub> | 0.46<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.58<br><sub>[0.01]</sub> | 0.33<br><sub>[0.03]</sub> | 0.29<br><sub>[0.04]</sub> | 0.63<br><sub>[0.03]</sub> | 0.44<br><sub>[0.03]</sub> |
| DEC | TF+ToD | 0.71<br><sub>[0.03]</sub> | 0.41<br><sub>[0.05]</sub> | 0.83<br><sub>[0.01]</sub> | 0.50<br><sub>[0.02]</sub> | 0.75<br><sub>[0.03]</sub> | 0.51<br><sub>[0.03]</sub> | 0.76<br><sub>[0.01]</sub> | 0.47<br><sub>[0.02]</sub> |
| DEC | BERT | *0.75* | *0.51* | *0.87* | 0.60 | 0.36 | 0.33 | 0.66 | 0.48 |
| DEC | BERT+ToD | 0.71 | 0.48 | 0.84 | 0.56 | **0.84** | **0.61** | **0.80** | **0.55** |

**$K = 100$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.74**<br><sub>[0.02]</sub> | **0.53**<br><sub>[0.02]</sub> | **0.84**<br><sub>[0.02]</sub> | **0.57**<br><sub>[0.04]</sub> | **0.79**<br><sub>[0.03]</sub> | **0.59**<br><sub>[0.04]</sub> | **0.79**<br><sub>[0.01]</sub> | **0.56**<br><sub>[0.02]</sub> |


### L1 all classes; $m=1$

**$K = 20$ clusters (20 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.66<br><sub>[0.01]</sub> | *0.32*<br><sub>[0.01]</sub> | 0.83<br><sub>[0.03]</sub> | 0.43<br><sub>[0.04]</sub> | 0.27<br><sub>[0.03]</sub> | 0.25<br><sub>[0.02]</sub> | 0.59<br><sub>[0.02]</sub> | *0.33*<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.56<br><sub>[0.11]</sub> | 0.23<br><sub>[0.04]</sub> | 0.78<br><sub>[0.02]</sub> | 0.28<br><sub>[0.03]</sub> | 0.75<br><sub>[0.00]</sub> | 0.42<br><sub>[0.02]</sub> | 0.69<br><sub>[0.03]</sub> | 0.31<br><sub>[0.02]</sub> |
| k-means | TF | *0.74*<br><sub>[0.03]</sub> | *0.34*<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.03]</sub> | 0.31<br><sub>[0.02]</sub> | 0.24<br><sub>[0.01]</sub> | 0.64<br><sub>[0.01]</sub> | *0.33*<br><sub>[0.02]</sub> |
| k-means | TF+ToD | 0.67<br><sub>[0.05]</sub> | 0.25<br><sub>[0.03]</sub> | 0.78<br><sub>[0.02]</sub> | 0.28<br><sub>[0.02]</sub> | *0.78*<br><sub>[0.04]</sub> | *0.47*<br><sub>[0.04]</sub> | *0.74*<br><sub>[0.03]</sub> | *0.33*<br><sub>[0.02]</sub> |
| k-means | BERT | 0.71 | 0.30 | **0.88** | **0.47** | 0.28 | 0.28 | 0.62 | **0.35** |
| k-means | BERT+ToD | 0.69 | 0.26 | 0.82 | 0.29 | *0.77* | **0.50** | **0.76** | **0.35** |
| SCAN | TF | *0.73*<br><sub>[0.02]</sub> | **0.34**<br><sub>[0.03]</sub> | 0.80<br><sub>[0.05]</sub> | 0.40<br><sub>[0.03]</sub> | 0.32<br><sub>[0.03]</sub> | 0.23<br><sub>[0.04]</sub> | 0.62<br><sub>[0.01]</sub> | *0.33*<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | 0.69<br><sub>[0.03]</sub> | 0.31<br><sub>[0.04]</sub> | 0.79<br><sub>[0.05]</sub> | 0.31<br><sub>[0.04]</sub> | 0.60<br><sub>[0.07]</sub> | 0.35<br><sub>[0.05]</sub> | 0.69<br><sub>[0.05]</sub> | *0.32*<br><sub>[0.03]</sub> |
| SCAN (unweighted ToD) | TF+ToD | 0.71<br><sub>[0.04]</sub> | **0.34**<br><sub>[0.03]</sub> | 0.78<br><sub>[0.04]</sub> | 0.39<br><sub>[0.03]</sub> | 0.38<br><sub>[0.03]</sub> | 0.29<br><sub>[0.04]</sub> | 0.62<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.03]</sub> |
| SCAN | BERT | 0.60 | 0.24 | *0.86* | 0.43 | 0.28 | 0.16 | 0.58 | 0.28 |
| SCAN (unweighted ToD) | BERT+ToD | **0.76** | *0.34* | 0.73 | 0.39 | 0.25 | 0.23 | 0.58 | 0.32 |
| SCAN | Sensor+ToD hist. | 0.63<br><sub>[0.03]</sub> | 0.28<br><sub>[0.06]</sub> | 0.80<br><sub>[0.05]</sub> | 0.35<br><sub>[0.01]</sub> | 0.65<br><sub>[0.08]</sub> | 0.38<br><sub>[0.07]</sub> | 0.69<br><sub>[0.01]</sub> | *0.34*<br><sub>[0.03]</sub> |
| DEC | TF | 0.71<br><sub>[0.04]</sub> | 0.29<br><sub>[0.04]</sub> | 0.84<br><sub>[0.05]</sub> | 0.39<br><sub>[0.06]</sub> | 0.29<br><sub>[0.02]</sub> | 0.23<br><sub>[0.02]</sub> | 0.61<br><sub>[0.02]</sub> | 0.30<br><sub>[0.02]</sub> |
| DEC | TF+ToD | 0.68<br><sub>[0.04]</sub> | 0.27<br><sub>[0.03]</sub> | 0.79<br><sub>[0.03]</sub> | 0.27<br><sub>[0.02]</sub> | **0.79**<br><sub>[0.03]</sub> | *0.47*<br><sub>[0.02]</sub> | **0.76**<br><sub>[0.02]</sub> | *0.34*<br><sub>[0.02]</sub> |
| DEC | BERT | 0.71 | 0.27 | **0.88** | 0.43 | 0.32 | 0.28 | 0.63 | *0.33* |
| DEC | BERT+ToD | 0.68 | 0.21 | 0.82 | 0.32 | *0.77* | *0.48* | *0.76* | *0.34* |

**$K = 50$ clusters (50 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.73*<br><sub>[0.02]</sub> | 0.37<br><sub>[0.04]</sub> | 0.83<br><sub>[0.03]</sub> | 0.43<br><sub>[0.04]</sub> | 0.28<br><sub>[0.02]</sub> | 0.25<br><sub>[0.01]</sub> | 0.62<br><sub>[0.02]</sub> | 0.35<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.70<br><sub>[0.03]</sub> | 0.31<br><sub>[0.02]</sub> | 0.79<br><sub>[0.03]</sub> | 0.39<br><sub>[0.04]</sub> | 0.73<br><sub>[0.03]</sub> | 0.48<br><sub>[0.03]</sub> | 0.74<br><sub>[0.01]</sub> | 0.39<br><sub>[0.02]</sub> |
| k-means | TF | *0.75*<br><sub>[0.02]</sub> | 0.37<br><sub>[0.02]</sub> | *0.86*<br><sub>[0.01]</sub> | **0.50**<br><sub>[0.02]</sub> | 0.35<br><sub>[0.03]</sub> | 0.31<br><sub>[0.03]</sub> | 0.65<br><sub>[0.02]</sub> | 0.39<br><sub>[0.01]</sub> |
| k-means | TF+ToD | 0.73<br><sub>[0.02]</sub> | 0.35<br><sub>[0.02]</sub> | 0.85<br><sub>[0.01]</sub> | 0.40<br><sub>[0.03]</sub> | 0.78<br><sub>[0.02]</sub> | 0.50<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | *0.42*<br><sub>[0.01]</sub> |
| k-means | BERT | **0.76** | **0.44** | **0.88** | 0.47 | 0.27 | 0.26 | 0.64 | 0.39 |
| k-means | BERT+ToD | *0.75* | 0.37 | *0.86* | 0.41 | 0.80 | *0.54* | **0.80** | **0.44** |
| SCAN | TF | *0.74*<br><sub>[0.03]</sub> | 0.35<br><sub>[0.05]</sub> | 0.84<br><sub>[0.02]</sub> | 0.44<br><sub>[0.03]</sub> | 0.34<br><sub>[0.01]</sub> | 0.31<br><sub>[0.02]</sub> | 0.64<br><sub>[0.00]</sub> | 0.37<br><sub>[0.01]</sub> |
| SCAN | TF+ToD | *0.75*<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> | 0.83<br><sub>[0.02]</sub> | 0.40<br><sub>[0.02]</sub> | 0.78<br><sub>[0.02]</sub> | 0.48<br><sub>[0.01]</sub> | *0.79*<br><sub>[0.01]</sub> | *0.42*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.75*<br><sub>[0.02]</sub> | 0.38<br><sub>[0.04]</sub> | 0.84<br><sub>[0.03]</sub> | 0.41<br><sub>[0.04]</sub> | 0.59<br><sub>[0.06]</sub> | 0.43<br><sub>[0.03]</sub> | 0.72<br><sub>[0.03]</sub> | 0.41<br><sub>[0.03]</sub> |
| SCAN | BERT | 0.70 | 0.27 | *0.87* | 0.43 | 0.32 | 0.30 | 0.63 | 0.33 |
| SCAN (unweighted ToD) | BERT+ToD | *0.75* | 0.37 | 0.80 | 0.44 | 0.38 | 0.33 | 0.64 | 0.38 |
| SCAN | Sensor+ToD hist. | 0.70<br><sub>[0.01]</sub> | 0.34<br><sub>[0.02]</sub> | 0.83<br><sub>[0.03]</sub> | 0.39<br><sub>[0.05]</sub> | 0.73<br><sub>[0.06]</sub> | 0.46<br><sub>[0.07]</sub> | 0.75<br><sub>[0.02]</sub> | 0.40<br><sub>[0.03]</sub> |
| DEC | TF | 0.70<br><sub>[0.07]</sub> | 0.34<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.41<br><sub>[0.01]</sub> | 0.33<br><sub>[0.03]</sub> | 0.26<br><sub>[0.03]</sub> | 0.63<br><sub>[0.03]</sub> | 0.34<br><sub>[0.02]</sub> |
| DEC | TF+ToD | 0.71<br><sub>[0.03]</sub> | 0.30<br><sub>[0.03]</sub> | 0.83<br><sub>[0.01]</sub> | 0.36<br><sub>[0.03]</sub> | 0.75<br><sub>[0.03]</sub> | 0.47<br><sub>[0.03]</sub> | 0.76<br><sub>[0.01]</sub> | 0.38<br><sub>[0.01]</sub> |
| DEC | BERT | *0.75* | 0.37 | *0.87* | 0.42 | 0.36 | 0.30 | 0.66 | 0.37 |
| DEC | BERT+ToD | 0.71 | 0.36 | 0.84 | 0.39 | **0.84** | **0.57** | **0.80** | **0.44** |

**$K = 100$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.74**<br><sub>[0.02]</sub> | **0.40**<br><sub>[0.02]</sub> | **0.84**<br><sub>[0.02]</sub> | **0.43**<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.03]</sub> | **0.54**<br><sub>[0.03]</sub> | **0.79**<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.02]</sub> |


### L2 all classes; $m=1$

**$K = 20$ clusters (20 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.67<br><sub>[0.01]</sub> | *0.35*<br><sub>[0.01]</sub> | 0.84<br><sub>[0.03]</sub> | 0.47<br><sub>[0.05]</sub> | 0.72<br><sub>[0.02]</sub> | 0.29<br><sub>[0.01]</sub> | 0.74<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> |
| k-means | Sensor+ToD hist. | 0.57<br><sub>[0.11]</sub> | 0.24<br><sub>[0.03]</sub> | 0.78<br><sub>[0.02]</sub> | 0.32<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | **0.41**<br><sub>[0.01]</sub> | 0.72<br><sub>[0.03]</sub> | 0.33<br><sub>[0.01]</sub> |
| k-means | TF | *0.75*<br><sub>[0.03]</sub> | *0.35*<br><sub>[0.04]</sub> | *0.88*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.04]</sub> | 0.72<br><sub>[0.02]</sub> | 0.28<br><sub>[0.02]</sub> | *0.78*<br><sub>[0.01]</sub> | *0.37*<br><sub>[0.03]</sub> |
| k-means | TF+ToD | 0.68<br><sub>[0.05]</sub> | 0.27<br><sub>[0.02]</sub> | 0.79<br><sub>[0.02]</sub> | 0.32<br><sub>[0.02]</sub> | *0.83*<br><sub>[0.02]</sub> | *0.39*<br><sub>[0.02]</sub> | *0.76*<br><sub>[0.02]</sub> | 0.33<br><sub>[0.02]</sub> |
| k-means | BERT | 0.72 | 0.32 | **0.89** | **0.54** | 0.69 | 0.35 | *0.76* | **0.40** |
| k-means | BERT+ToD | 0.70 | 0.31 | 0.84 | 0.37 | **0.84** | *0.40* | **0.79** | 0.36 |
| SCAN | TF | *0.75*<br><sub>[0.02]</sub> | **0.37**<br><sub>[0.03]</sub> | 0.80<br><sub>[0.06]</sub> | 0.43<br><sub>[0.04]</sub> | 0.74<br><sub>[0.04]</sub> | 0.30<br><sub>[0.02]</sub> | 0.76<br><sub>[0.02]</sub> | 0.37<br><sub>[0.02]</sub> |
| SCAN | TF+ToD | 0.70<br><sub>[0.03]</sub> | 0.33<br><sub>[0.06]</sub> | 0.79<br><sub>[0.06]</sub> | 0.33<br><sub>[0.07]</sub> | 0.69<br><sub>[0.07]</sub> | 0.32<br><sub>[0.05]</sub> | 0.73<br><sub>[0.05]</sub> | 0.33<br><sub>[0.05]</sub> |
| SCAN (unweighted ToD) | TF+ToD | 0.72<br><sub>[0.04]</sub> | *0.36*<br><sub>[0.02]</sub> | 0.77<br><sub>[0.04]</sub> | 0.42<br><sub>[0.02]</sub> | 0.76<br><sub>[0.05]</sub> | 0.34<br><sub>[0.05]</sub> | 0.75<br><sub>[0.02]</sub> | *0.38*<br><sub>[0.02]</sub> |
| SCAN | BERT | 0.64 | 0.33 | *0.86* | 0.47 | 0.75 | 0.27 | 0.75 | 0.36 |
| SCAN (unweighted ToD) | BERT+ToD | **0.76** | 0.33 | 0.73 | 0.42 | 0.71 | 0.29 | 0.73 | 0.35 |
| SCAN | Sensor+ToD hist. | 0.65<br><sub>[0.02]</sub> | 0.33<br><sub>[0.06]</sub> | 0.80<br><sub>[0.06]</sub> | 0.38<br><sub>[0.02]</sub> | 0.76<br><sub>[0.07]</sub> | 0.35<br><sub>[0.07]</sub> | 0.74<br><sub>[0.02]</sub> | 0.35<br><sub>[0.04]</sub> |
| DEC | TF | 0.72<br><sub>[0.04]</sub> | 0.29<br><sub>[0.04]</sub> | 0.84<br><sub>[0.06]</sub> | 0.41<br><sub>[0.07]</sub> | 0.74<br><sub>[0.03]</sub> | 0.31<br><sub>[0.03]</sub> | *0.77*<br><sub>[0.03]</sub> | 0.34<br><sub>[0.03]</sub> |
| DEC | TF+ToD | 0.70<br><sub>[0.04]</sub> | 0.31<br><sub>[0.04]</sub> | 0.80<br><sub>[0.03]</sub> | 0.33<br><sub>[0.02]</sub> | **0.84**<br><sub>[0.02]</sub> | *0.40*<br><sub>[0.01]</sub> | *0.78*<br><sub>[0.02]</sub> | 0.35<br><sub>[0.01]</sub> |
| DEC | BERT | 0.72 | 0.31 | **0.88** | 0.47 | 0.76 | 0.32 | **0.79** | 0.36 |
| DEC | BERT+ToD | 0.68 | 0.23 | 0.82 | 0.36 | **0.84** | *0.40* | *0.78* | 0.33 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.74<br><sub>[0.02]</sub> | 0.37<br><sub>[0.03]</sub> | 0.83<br><sub>[0.03]</sub> | 0.48<br><sub>[0.05]</sub> | 0.69<br><sub>[0.02]</sub> | 0.28<br><sub>[0.01]</sub> | 0.75<br><sub>[0.02]</sub> | 0.38<br><sub>[0.03]</sub> |
| k-means | Sensor+ToD hist. | 0.71<br><sub>[0.03]</sub> | 0.33<br><sub>[0.02]</sub> | 0.79<br><sub>[0.04]</sub> | 0.42<br><sub>[0.06]</sub> | 0.80<br><sub>[0.02]</sub> | 0.42<br><sub>[0.02]</sub> | 0.77<br><sub>[0.01]</sub> | 0.39<br><sub>[0.03]</sub> |
| k-means | TF | *0.76*<br><sub>[0.02]</sub> | 0.37<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.01]</sub> | **0.55**<br><sub>[0.02]</sub> | 0.77<br><sub>[0.01]</sub> | 0.35<br><sub>[0.04]</sub> | *0.80*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.01]</sub> |
| k-means | TF+ToD | 0.74<br><sub>[0.01]</sub> | 0.38<br><sub>[0.02]</sub> | 0.85<br><sub>[0.01]</sub> | 0.45<br><sub>[0.02]</sub> | *0.84*<br><sub>[0.02]</sub> | 0.41<br><sub>[0.01]</sub> | *0.81*<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.00]</sub> |
| k-means | BERT | **0.77** | *0.39* | **0.89** | 0.52 | 0.69 | 0.30 | 0.78 | *0.41* |
| k-means | BERT+ToD | *0.75* | *0.40* | 0.86 | 0.46 | *0.85* | *0.44* | **0.82** | **0.43** |
| SCAN | TF | *0.75*<br><sub>[0.02]</sub> | 0.37<br><sub>[0.06]</sub> | 0.84<br><sub>[0.02]</sub> | 0.48<br><sub>[0.05]</sub> | 0.76<br><sub>[0.02]</sub> | 0.35<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | 0.40<br><sub>[0.02]</sub> |
| SCAN | TF+ToD | *0.76*<br><sub>[0.02]</sub> | **0.41**<br><sub>[0.02]</sub> | 0.83<br><sub>[0.03]</sub> | 0.43<br><sub>[0.02]</sub> | 0.84<br><sub>[0.02]</sub> | *0.44*<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.75*<br><sub>[0.02]</sub> | *0.40*<br><sub>[0.01]</sub> | 0.84<br><sub>[0.03]</sub> | 0.45<br><sub>[0.04]</sub> | 0.79<br><sub>[0.03]</sub> | 0.39<br><sub>[0.01]</sub> | *0.79*<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.01]</sub> |
| SCAN | BERT | 0.72 | 0.31 | *0.87* | 0.48 | 0.76 | 0.35 | 0.78 | 0.38 |
| SCAN (unweighted ToD) | BERT+ToD | *0.75* | *0.39* | 0.79 | 0.46 | 0.75 | 0.34 | 0.77 | 0.39 |
| SCAN | Sensor+ToD hist. | 0.71<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> | 0.83<br><sub>[0.03]</sub> | 0.42<br><sub>[0.05]</sub> | 0.78<br><sub>[0.05]</sub> | 0.38<br><sub>[0.05]</sub> | 0.77<br><sub>[0.02]</sub> | 0.39<br><sub>[0.02]</sub> |
| DEC | TF | 0.71<br><sub>[0.06]</sub> | 0.34<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.44<br><sub>[0.02]</sub> | 0.76<br><sub>[0.02]</sub> | 0.32<br><sub>[0.02]</sub> | 0.78<br><sub>[0.03]</sub> | 0.37<br><sub>[0.02]</sub> |
| DEC | TF+ToD | 0.72<br><sub>[0.03]</sub> | 0.33<br><sub>[0.04]</sub> | 0.84<br><sub>[0.02]</sub> | 0.39<br><sub>[0.03]</sub> | 0.81<br><sub>[0.02]</sub> | 0.40<br><sub>[0.02]</sub> | 0.79<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> |
| DEC | BERT | *0.76* | *0.40* | *0.87* | 0.46 | 0.78 | 0.36 | *0.80* | *0.41* |
| DEC | BERT+ToD | 0.71 | *0.40* | 0.85 | 0.42 | **0.87** | **0.46** | *0.81* | **0.43** |

**$K = 100$ clusters (100 labels)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.75**<br><sub>[0.02]</sub> | **0.43**<br><sub>[0.01]</sub> | **0.84**<br><sub>[0.02]</sub> | **0.47**<br><sub>[0.04]</sub> | **0.85**<br><sub>[0.03]</sub> | **0.48**<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.02]</sub> |

