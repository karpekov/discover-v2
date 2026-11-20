### L1 masked (primary); $m=all$

**$K = 20$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.78*<br><sub>[0.00]</sub> | 0.52<br><sub>[0.00]</sub> | *0.87*<br><sub>[0.00]</sub> | *0.61*<br><sub>[0.02]</sub> | 0.31<br><sub>[0.00]</sub> | 0.30<br><sub>[0.01]</sub> | – | – | 0.65 | 0.48 |
| k-means | Sensor+ToD hist. | 0.70<br><sub>[0.01]</sub> | 0.38<br><sub>[0.02]</sub> | 0.81<br><sub>[0.01]</sub> | 0.42<br><sub>[0.03]</sub> | *0.78*<br><sub>[0.00]</sub> | 0.52<br><sub>[0.02]</sub> | – | – | *0.76* | 0.44 |
| k-means | TF | **0.79**<br><sub>[0.01]</sub> | *0.56*<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.00]</sub> | *0.62*<br><sub>[0.02]</sub> | 0.33<br><sub>[0.01]</sub> | 0.32<br><sub>[0.01]</sub> | – | – | 0.67 | *0.50* |
| k-means | TF+ToD | 0.70<br><sub>[0.01]</sub> | 0.37<br><sub>[0.04]</sub> | 0.80<br><sub>[0.01]</sub> | 0.37<br><sub>[0.03]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.02]</sub> | – | – | *0.76* | 0.42 |
| k-means | BERT | **0.79** | **0.59** | **0.88** | **0.63** | 0.32 | 0.31 | – | – | 0.66 | **0.51** |
| k-means | BERT+ToD | 0.71 | 0.35 | 0.83 | 0.48 | **0.80** | **0.57** | – | – | **0.78** | 0.46 |
| SCAN | TF | *0.76*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.02]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.55<br><sub>[0.05]</sub> | 0.34<br><sub>[0.01]</sub> | 0.29<br><sub>[0.01]</sub> | – | – | 0.66 | 0.43 |
| SCAN | TF+ToD | 0.74<br><sub>[0.02]</sub> | 0.46<br><sub>[0.02]</sub> | 0.83<br><sub>[0.01]</sub> | 0.49<br><sub>[0.02]</sub> | 0.72<br><sub>[0.01]</sub> | 0.45<br><sub>[0.03]</sub> | **0.75**<br><sub>[0.01]</sub> | **0.50**<br><sub>[0.05]</sub> | *0.76*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.01]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.78*<br><sub>[0.01]</sub> | 0.49<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.56<br><sub>[0.07]</sub> | 0.42<br><sub>[0.04]</sub> | 0.37<br><sub>[0.04]</sub> | – | – | 0.68 | 0.47 |
| SCAN | BERT | 0.76 | 0.42 | *0.85* | 0.55 | 0.34 | 0.31 | – | – | 0.65 | 0.42 |
| SCAN (unweighted ToD) | BERT+ToD | *0.77* | 0.46 | *0.86* | 0.58 | 0.33 | 0.32 | – | – | 0.65 | 0.45 |
| SCAN | Sensor+ToD hist. | 0.74<br><sub>[0.02]</sub> | 0.41<br><sub>[0.05]</sub> | 0.84<br><sub>[0.01]</sub> | 0.50<br><sub>[0.03]</sub> | 0.69<br><sub>[0.08]</sub> | 0.43<br><sub>[0.05]</sub> | – | – | *0.76* | 0.45 |
| DEC | TF | 0.76<br><sub>[0.01]</sub> | 0.44<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.58<br><sub>[0.01]</sub> | 0.31<br><sub>[0.01]</sub> | 0.25<br><sub>[0.05]</sub> | – | – | 0.65 | 0.42 |
| DEC | TF+ToD | 0.70<br><sub>[0.01]</sub> | 0.34<br><sub>[0.01]</sub> | 0.81<br><sub>[0.01]</sub> | 0.38<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.52<br><sub>[0.01]</sub> | – | – | *0.77* | 0.41 |
| DEC | BERT | 0.76 | 0.45 | *0.85* | 0.53 | 0.31 | 0.28 | – | – | 0.64 | 0.42 |
| DEC | BERT+ToD | 0.69 | 0.32 | 0.81 | 0.36 | **0.80** | 0.53 | – | – | *0.76* | 0.40 |

**$K = 50$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.80**<br><sub>[0.00]</sub> | *0.61*<br><sub>[0.01]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.61<br><sub>[0.00]</sub> | 0.35<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> | – | – | 0.68 | 0.52 |
| k-means | Sensor+ToD hist. | *0.79*<br><sub>[0.01]</sub> | 0.56<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.00]</sub> | 0.60<br><sub>[0.01]</sub> | *0.80*<br><sub>[0.00]</sub> | **0.61**<br><sub>[0.00]</sub> | – | – | *0.82* | **0.59** |
| k-means | TF | *0.80*<br><sub>[0.01]</sub> | *0.60*<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.00]</sub> | *0.61*<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> | 0.35<br><sub>[0.01]</sub> | – | – | 0.68 | 0.52 |
| k-means | TF+ToD | *0.80*<br><sub>[0.01]</sub> | 0.55<br><sub>[0.00]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.58<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.00]</sub> | *0.59*<br><sub>[0.02]</sub> | – | – | **0.82** | *0.57* |
| k-means | BERT | *0.78* | 0.56 | **0.88** | *0.63* | 0.37 | 0.37 | – | – | 0.68 | 0.52 |
| k-means | BERT+ToD | *0.79* | 0.59 | *0.86* | 0.56 | **0.82** | *0.59* | – | – | **0.82** | *0.58* |
| SCAN | TF | *0.79*<br><sub>[0.01]</sub> | 0.59<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.00]</sub> | *0.61*<br><sub>[0.01]</sub> | 0.38<br><sub>[0.01]</sub> | 0.34<br><sub>[0.03]</sub> | – | – | 0.68 | 0.52 |
| SCAN | TF+ToD | *0.78*<br><sub>[0.02]</sub> | 0.51<br><sub>[0.04]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.57<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.01]</sub> | *0.59*<br><sub>[0.01]</sub> | **0.81**<br><sub>[0.01]</sub> | **0.62**<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | *0.57*<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.80**<br><sub>[0.01]</sub> | 0.58<br><sub>[0.04]</sub> | **0.88**<br><sub>[0.00]</sub> | *0.62*<br><sub>[0.01]</sub> | 0.67<br><sub>[0.09]</sub> | 0.53<br><sub>[0.03]</sub> | – | – | 0.78 | *0.58* |
| SCAN | BERT | 0.77 | 0.54 | **0.88** | **0.64** | 0.36 | 0.36 | – | – | 0.67 | 0.51 |
| SCAN (unweighted ToD) | BERT+ToD | **0.81** | **0.62** | *0.87* | 0.59 | 0.41 | 0.38 | – | – | 0.70 | 0.53 |
| SCAN | Sensor+ToD hist. | 0.77<br><sub>[0.02]</sub> | 0.51<br><sub>[0.05]</sub> | *0.85*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.04]</sub> | 0.79<br><sub>[0.02]</sub> | 0.58<br><sub>[0.02]</sub> | – | – | *0.80* | 0.54 |
| DEC | TF | 0.77<br><sub>[0.01]</sub> | 0.50<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.57<br><sub>[0.02]</sub> | 0.33<br><sub>[0.01]</sub> | 0.29<br><sub>[0.03]</sub> | – | – | 0.66 | 0.46 |
| DEC | TF+ToD | 0.76<br><sub>[0.01]</sub> | 0.45<br><sub>[0.02]</sub> | 0.85<br><sub>[0.02]</sub> | 0.48<br><sub>[0.04]</sub> | 0.79<br><sub>[0.01]</sub> | 0.54<br><sub>[0.01]</sub> | – | – | *0.80* | 0.49 |
| DEC | BERT | *0.78* | 0.56 | **0.88** | *0.61* | 0.33 | 0.30 | – | – | 0.67 | 0.49 |
| DEC | BERT+ToD | 0.77 | 0.48 | 0.85 | 0.51 | **0.81** | 0.58 | – | – | *0.81* | 0.52 |

**$K = 100$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.81**<br><sub>[0.01]</sub> | **0.61**<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.01]</sub> | **0.62**<br><sub>[0.02]</sub> | **0.83**<br><sub>[0.00]</sub> | **0.64**<br><sub>[0.01]</sub> | – | – | **0.84** | **0.62** |


### L1 all classes; $m=all$

**$K = 20$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.78*<br><sub>[0.00]</sub> | 0.38<br><sub>[0.00]</sub> | *0.87*<br><sub>[0.00]</sub> | *0.43*<br><sub>[0.02]</sub> | 0.31<br><sub>[0.00]</sub> | 0.27<br><sub>[0.01]</sub> | – | – | 0.65 | *0.36* |
| k-means | Sensor+ToD hist. | 0.70<br><sub>[0.01]</sub> | 0.28<br><sub>[0.02]</sub> | 0.81<br><sub>[0.01]</sub> | 0.29<br><sub>[0.02]</sub> | *0.78*<br><sub>[0.00]</sub> | 0.47<br><sub>[0.02]</sub> | – | – | *0.76* | 0.35 |
| k-means | TF | **0.79**<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.00]</sub> | *0.43*<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> | 0.29<br><sub>[0.01]</sub> | – | – | 0.67 | *0.38* |
| k-means | TF+ToD | 0.70<br><sub>[0.01]</sub> | 0.27<br><sub>[0.03]</sub> | 0.80<br><sub>[0.01]</sub> | 0.26<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.48<br><sub>[0.02]</sub> | – | – | *0.76* | 0.34 |
| k-means | BERT | **0.79** | **0.43** | **0.88** | **0.44** | 0.32 | 0.29 | – | – | 0.66 | **0.39** |
| k-means | BERT+ToD | 0.71 | 0.26 | 0.83 | 0.33 | **0.80** | **0.51** | – | – | **0.78** | *0.37* |
| SCAN | TF | *0.76*<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.39<br><sub>[0.03]</sub> | 0.34<br><sub>[0.01]</sub> | 0.27<br><sub>[0.01]</sub> | – | – | 0.66 | 0.33 |
| SCAN | TF+ToD | 0.74<br><sub>[0.02]</sub> | 0.34<br><sub>[0.02]</sub> | 0.83<br><sub>[0.01]</sub> | 0.34<br><sub>[0.01]</sub> | 0.72<br><sub>[0.01]</sub> | 0.41<br><sub>[0.03]</sub> | **0.75**<br><sub>[0.01]</sub> | **0.23**<br><sub>[0.01]</sub> | *0.76*<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.78*<br><sub>[0.01]</sub> | 0.36<br><sub>[0.02]</sub> | *0.86*<br><sub>[0.02]</sub> | 0.39<br><sub>[0.05]</sub> | 0.42<br><sub>[0.04]</sub> | 0.33<br><sub>[0.03]</sub> | – | – | 0.68 | *0.36* |
| SCAN | BERT | 0.76 | 0.31 | *0.85* | *0.43* | 0.34 | 0.28 | – | – | 0.65 | 0.34 |
| SCAN (unweighted ToD) | BERT+ToD | *0.77* | 0.34 | *0.86* | **0.44** | 0.33 | 0.29 | – | – | 0.65 | *0.36* |
| SCAN | Sensor+ToD hist. | 0.74<br><sub>[0.02]</sub> | 0.30<br><sub>[0.04]</sub> | 0.84<br><sub>[0.01]</sub> | 0.35<br><sub>[0.02]</sub> | 0.69<br><sub>[0.08]</sub> | 0.39<br><sub>[0.05]</sub> | – | – | *0.76* | 0.35 |
| DEC | TF | 0.76<br><sub>[0.01]</sub> | 0.32<br><sub>[0.01]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.40<br><sub>[0.01]</sub> | 0.31<br><sub>[0.01]</sub> | 0.22<br><sub>[0.05]</sub> | – | – | 0.65 | 0.32 |
| DEC | TF+ToD | 0.70<br><sub>[0.01]</sub> | 0.25<br><sub>[0.01]</sub> | 0.81<br><sub>[0.01]</sub> | 0.26<br><sub>[0.02]</sub> | *0.79*<br><sub>[0.01]</sub> | 0.47<br><sub>[0.01]</sub> | – | – | *0.77* | 0.33 |
| DEC | BERT | 0.76 | 0.33 | *0.85* | 0.37 | 0.31 | 0.25 | – | – | 0.64 | 0.32 |
| DEC | BERT+ToD | 0.69 | 0.24 | 0.81 | 0.25 | **0.80** | 0.48 | – | – | *0.76* | 0.32 |

**$K = 50$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.80**<br><sub>[0.00]</sub> | **0.50**<br><sub>[0.00]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.43<br><sub>[0.00]</sub> | 0.35<br><sub>[0.01]</sub> | 0.30<br><sub>[0.01]</sub> | – | – | 0.68 | 0.41 |
| k-means | Sensor+ToD hist. | *0.79*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.00]</sub> | 0.42<br><sub>[0.01]</sub> | *0.80*<br><sub>[0.00]</sub> | **0.56**<br><sub>[0.00]</sub> | – | – | *0.82* | **0.46** |
| k-means | TF | *0.80*<br><sub>[0.01]</sub> | 0.47<br><sub>[0.04]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.43<br><sub>[0.01]</sub> | 0.37<br><sub>[0.01]</sub> | 0.32<br><sub>[0.01]</sub> | – | – | 0.68 | 0.41 |
| k-means | TF+ToD | *0.80*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.00]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.41<br><sub>[0.02]</sub> | *0.81*<br><sub>[0.00]</sub> | *0.54*<br><sub>[0.02]</sub> | – | – | **0.82** | *0.45* |
| k-means | BERT | *0.78* | *0.47* | **0.88** | *0.47* | 0.37 | 0.33 | – | – | 0.68 | 0.42 |
| k-means | BERT+ToD | *0.79* | 0.43 | *0.86* | 0.40 | **0.82** | **0.56** | – | – | **0.82** | **0.46** |
| SCAN | TF | *0.79*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.43<br><sub>[0.00]</sub> | 0.38<br><sub>[0.01]</sub> | 0.31<br><sub>[0.03]</sub> | – | – | 0.68 | 0.39 |
| SCAN | TF+ToD | *0.78*<br><sub>[0.02]</sub> | 0.38<br><sub>[0.03]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.40<br><sub>[0.02]</sub> | **0.81**<br><sub>[0.01]</sub> | *0.53*<br><sub>[0.01]</sub> | **0.81**<br><sub>[0.01]</sub> | **0.38**<br><sub>[0.02]</sub> | *0.82*<br><sub>[0.01]</sub> | 0.42<br><sub>[0.01]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.80**<br><sub>[0.01]</sub> | 0.42<br><sub>[0.03]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.43<br><sub>[0.01]</sub> | 0.67<br><sub>[0.09]</sub> | 0.48<br><sub>[0.03]</sub> | – | – | 0.78 | *0.45* |
| SCAN | BERT | 0.77 | 0.40 | **0.88** | **0.48** | 0.36 | 0.33 | – | – | 0.67 | 0.40 |
| SCAN (unweighted ToD) | BERT+ToD | **0.81** | 0.46 | *0.87* | 0.43 | 0.41 | 0.35 | – | – | 0.70 | 0.41 |
| SCAN | Sensor+ToD hist. | 0.77<br><sub>[0.02]</sub> | 0.37<br><sub>[0.04]</sub> | *0.85*<br><sub>[0.01]</sub> | 0.38<br><sub>[0.03]</sub> | 0.79<br><sub>[0.02]</sub> | 0.52<br><sub>[0.02]</sub> | – | – | *0.80* | 0.43 |
| DEC | TF | 0.77<br><sub>[0.01]</sub> | 0.38<br><sub>[0.03]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.40<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> | 0.26<br><sub>[0.03]</sub> | – | – | 0.66 | 0.35 |
| DEC | TF+ToD | 0.76<br><sub>[0.01]</sub> | 0.33<br><sub>[0.01]</sub> | 0.85<br><sub>[0.02]</sub> | 0.34<br><sub>[0.03]</sub> | 0.79<br><sub>[0.01]</sub> | 0.49<br><sub>[0.01]</sub> | – | – | *0.80* | 0.39 |
| DEC | BERT | *0.78* | 0.41 | **0.88** | 0.43 | 0.33 | 0.27 | – | – | 0.67 | 0.37 |
| DEC | BERT+ToD | 0.77 | 0.35 | 0.85 | 0.36 | **0.81** | *0.55* | – | – | *0.81* | 0.42 |

**$K = 100$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.81**<br><sub>[0.01]</sub> | **0.45**<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.01]</sub> | **0.83**<br><sub>[0.00]</sub> | **0.59**<br><sub>[0.02]</sub> | – | – | **0.84** | **0.49** |


### L2 all classes; $m=all$

**$K = 20$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | *0.79*<br><sub>[0.00]</sub> | 0.38<br><sub>[0.00]</sub> | *0.88*<br><sub>[0.00]</sub> | 0.48<br><sub>[0.01]</sub> | 0.75<br><sub>[0.01]</sub> | 0.31<br><sub>[0.01]</sub> | – | – | *0.80* | 0.39 |
| k-means | Sensor+ToD hist. | 0.72<br><sub>[0.01]</sub> | 0.30<br><sub>[0.02]</sub> | 0.81<br><sub>[0.01]</sub> | 0.34<br><sub>[0.02]</sub> | *0.84*<br><sub>[0.00]</sub> | *0.40*<br><sub>[0.01]</sub> | – | – | 0.79 | 0.35 |
| k-means | TF | **0.79**<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.03]</sub> | *0.88*<br><sub>[0.00]</sub> | *0.50*<br><sub>[0.01]</sub> | 0.76<br><sub>[0.01]</sub> | 0.33<br><sub>[0.00]</sub> | – | – | *0.81* | *0.42* |
| k-means | TF+ToD | 0.72<br><sub>[0.01]</sub> | 0.30<br><sub>[0.01]</sub> | 0.80<br><sub>[0.01]</sub> | 0.31<br><sub>[0.02]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.01]</sub> | – | – | *0.79* | 0.34 |
| k-means | BERT | **0.80** | **0.46** | **0.89** | *0.50* | 0.77 | 0.34 | – | – | **0.82** | **0.43** |
| k-means | BERT+ToD | 0.72 | 0.30 | 0.84 | 0.37 | **0.87** | **0.42** | – | – | *0.81* | 0.36 |
| SCAN | TF | 0.77<br><sub>[0.01]</sub> | 0.31<br><sub>[0.01]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.46<br><sub>[0.04]</sub> | 0.77<br><sub>[0.01]</sub> | 0.32<br><sub>[0.02]</sub> | – | – | *0.80* | 0.36 |
| SCAN | TF+ToD | 0.76<br><sub>[0.02]</sub> | 0.36<br><sub>[0.02]</sub> | 0.85<br><sub>[0.01]</sub> | 0.40<br><sub>[0.01]</sub> | 0.81<br><sub>[0.02]</sub> | 0.36<br><sub>[0.03]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.31**<br><sub>[0.05]</sub> | *0.81*<br><sub>[0.01]</sub> | 0.36<br><sub>[0.02]</sub> |
| SCAN (unweighted ToD) | TF+ToD | *0.79*<br><sub>[0.02]</sub> | 0.35<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.02]</sub> | 0.46<br><sub>[0.06]</sub> | 0.81<br><sub>[0.03]</sub> | 0.36<br><sub>[0.04]</sub> | – | – | **0.82** | 0.39 |
| SCAN | BERT | 0.76 | 0.30 | *0.86* | **0.52** | 0.77 | 0.31 | – | – | *0.80* | 0.38 |
| SCAN (unweighted ToD) | BERT+ToD | *0.77* | 0.32 | *0.88* | **0.53** | 0.79 | 0.35 | – | – | *0.81* | 0.40 |
| SCAN | Sensor+ToD hist. | 0.75<br><sub>[0.02]</sub> | 0.32<br><sub>[0.06]</sub> | 0.85<br><sub>[0.01]</sub> | 0.40<br><sub>[0.02]</sub> | 0.83<br><sub>[0.02]</sub> | 0.38<br><sub>[0.02]</sub> | – | – | *0.81* | 0.37 |
| DEC | TF | 0.76<br><sub>[0.01]</sub> | 0.31<br><sub>[0.02]</sub> | **0.89**<br><sub>[0.00]</sub> | 0.46<br><sub>[0.01]</sub> | 0.74<br><sub>[0.01]</sub> | 0.27<br><sub>[0.03]</sub> | – | – | *0.80* | 0.35 |
| DEC | TF+ToD | 0.71<br><sub>[0.02]</sub> | 0.29<br><sub>[0.02]</sub> | 0.82<br><sub>[0.02]</sub> | 0.33<br><sub>[0.02]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.41*<br><sub>[0.01]</sub> | – | – | *0.80* | 0.34 |
| DEC | BERT | 0.76 | 0.32 | *0.86* | 0.40 | 0.74 | 0.26 | – | – | 0.79 | 0.33 |
| DEC | BERT+ToD | 0.70 | 0.28 | 0.81 | 0.31 | **0.87** | **0.42** | – | – | 0.79 | 0.34 |

**$K = 50$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.81**<br><sub>[0.00]</sub> | *0.47*<br><sub>[0.00]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.49<br><sub>[0.00]</sub> | 0.78<br><sub>[0.00]</sub> | 0.34<br><sub>[0.00]</sub> | – | – | *0.82* | *0.44* |
| k-means | Sensor+ToD hist. | *0.80*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.00]</sub> | 0.47<br><sub>[0.01]</sub> | *0.86*<br><sub>[0.00]</sub> | **0.43**<br><sub>[0.00]</sub> | – | – | *0.84* | *0.45* |
| k-means | TF | *0.81*<br><sub>[0.00]</sub> | *0.47*<br><sub>[0.01]</sub> | **0.89**<br><sub>[0.00]</sub> | 0.49<br><sub>[0.01]</sub> | 0.79<br><sub>[0.01]</sub> | 0.35<br><sub>[0.01]</sub> | – | – | *0.83* | *0.44* |
| k-means | TF+ToD | *0.81*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.01]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.47<br><sub>[0.02]</sub> | *0.87*<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.01]</sub> | – | – | **0.85** | *0.44* |
| k-means | BERT | *0.78* | 0.43 | **0.89** | *0.53* | 0.81 | 0.38 | – | – | *0.83* | *0.45* |
| k-means | BERT+ToD | *0.80* | *0.47* | *0.87* | 0.45 | *0.87* | *0.42* | – | – | **0.85** | *0.45* |
| SCAN | TF | *0.80*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.04]</sub> | **0.88**<br><sub>[0.00]</sub> | 0.49<br><sub>[0.00]</sub> | 0.79<br><sub>[0.01]</sub> | 0.36<br><sub>[0.01]</sub> | – | – | *0.83* | *0.44* |
| SCAN | TF+ToD | *0.79*<br><sub>[0.02]</sub> | 0.39<br><sub>[0.04]</sub> | *0.87*<br><sub>[0.01]</sub> | 0.46<br><sub>[0.02]</sub> | **0.88**<br><sub>[0.01]</sub> | **0.44**<br><sub>[0.01]</sub> | **0.85**<br><sub>[0.01]</sub> | **0.45**<br><sub>[0.01]</sub> | **0.84**<br><sub>[0.01]</sub> | *0.43*<br><sub>[0.01]</sub> |
| SCAN (unweighted ToD) | TF+ToD | **0.81**<br><sub>[0.01]</sub> | 0.45<br><sub>[0.04]</sub> | **0.89**<br><sub>[0.00]</sub> | 0.49<br><sub>[0.01]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.42*<br><sub>[0.02]</sub> | – | – | **0.85** | **0.45** |
| SCAN | BERT | *0.78* | 0.42 | **0.89** | **0.53** | 0.80 | 0.37 | – | – | *0.82* | *0.44* |
| SCAN (unweighted ToD) | BERT+ToD | **0.81** | **0.49** | *0.88* | 0.48 | 0.81 | 0.38 | – | – | *0.83* | **0.45** |
| SCAN | Sensor+ToD hist. | *0.78*<br><sub>[0.01]</sub> | 0.40<br><sub>[0.05]</sub> | *0.86*<br><sub>[0.01]</sub> | 0.44<br><sub>[0.03]</sub> | *0.85*<br><sub>[0.02]</sub> | *0.41*<br><sub>[0.02]</sub> | – | – | *0.83* | 0.42 |
| DEC | TF | 0.78<br><sub>[0.01]</sub> | 0.37<br><sub>[0.02]</sub> | *0.88*<br><sub>[0.00]</sub> | 0.46<br><sub>[0.01]</sub> | 0.77<br><sub>[0.01]</sub> | 0.32<br><sub>[0.01]</sub> | – | – | 0.81 | 0.38 |
| DEC | TF+ToD | 0.77<br><sub>[0.01]</sub> | 0.36<br><sub>[0.01]</sub> | 0.86<br><sub>[0.01]</sub> | 0.39<br><sub>[0.01]</sub> | *0.85*<br><sub>[0.01]</sub> | *0.42*<br><sub>[0.01]</sub> | – | – | *0.83* | 0.39 |
| DEC | BERT | *0.79* | 0.43 | **0.89** | 0.48 | 0.79 | 0.35 | – | – | *0.82* | 0.42 |
| DEC | BERT+ToD | 0.78 | 0.36 | 0.85 | 0.41 | **0.88** | *0.43* | – | – | *0.84* | 0.40 |

**$K = 100$ clusters (all labeled windows vote)**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | orange W | orange M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCAN | TF+ToD | **0.82**<br><sub>[0.01]</sub> | **0.49**<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.01]</sub> | **0.49**<br><sub>[0.01]</sub> | **0.88**<br><sub>[0.01]</sub> | **0.46**<br><sub>[0.01]</sub> | – | – | **0.86** | **0.48** |

