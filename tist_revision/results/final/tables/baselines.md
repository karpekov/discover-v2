### Baselines, L1 masked (primary)

**budget = 100 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.65**<br><sub>[0.07]</sub> | **0.29**<br><sub>[0.06]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.36**<br><sub>[0.07]</sub> | **0.28**<br><sub>[0.03]</sub> | **0.18**<br><sub>[0.05]</sub> | **0.58**<br><sub>[0.03]</sub> | **0.28**<br><sub>[0.03]</sub> |
| TDOST | — | *0.63*<br><sub>[0.04]</sub> | 0.24<br><sub>[0.05]</sub> | 0.78<br><sub>[0.04]</sub> | 0.31<br><sub>[0.09]</sub> | *0.26*<br><sub>[0.05]</sub> | *0.17*<br><sub>[0.04]</sub> | *0.55*<br><sub>[0.03]</sub> | 0.24<br><sub>[0.04]</sub> |

**budget = 250 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.72**<br><sub>[0.02]</sub> | **0.38**<br><sub>[0.07]</sub> | **0.86**<br><sub>[0.02]</sub> | **0.55**<br><sub>[0.06]</sub> | **0.30**<br><sub>[0.03]</sub> | **0.20**<br><sub>[0.06]</sub> | **0.63**<br><sub>[0.01]</sub> | **0.38**<br><sub>[0.02]</sub> |
| TDOST | — | 0.67<br><sub>[0.03]</sub> | 0.30<br><sub>[0.07]</sub> | *0.83*<br><sub>[0.02]</sub> | 0.43<br><sub>[0.09]</sub> | **0.31**<br><sub>[0.04]</sub> | *0.19*<br><sub>[0.05]</sub> | *0.60*<br><sub>[0.02]</sub> | 0.31<br><sub>[0.04]</sub> |

**all training windows**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.81**<br><sub>[0.00]</sub> | **0.61**<br><sub>[0.01]</sub> | **0.90**<br><sub>[0.00]</sub> | **0.67**<br><sub>[0.00]</sub> | 0.41<br><sub>[0.01]</sub> | 0.40<br><sub>[0.00]</sub> | 0.70<br><sub>[0.00]</sub> | 0.56<br><sub>[0.00]</sub> |
| TDOST | — | *0.78*<br><sub>[0.01]</sub> | 0.54<br><sub>[0.02]</sub> | *0.89*<br><sub>[0.01]</sub> | *0.64*<br><sub>[0.01]</sub> | **0.87**<br><sub>[0.01]</sub> | **0.66**<br><sub>[0.01]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.61**<br><sub>[0.01]</sub> |


### Baselines, L1 all classes

**budget = 100 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.65**<br><sub>[0.07]</sub> | **0.22**<br><sub>[0.05]</sub> | **0.81**<br><sub>[0.02]</sub> | **0.25**<br><sub>[0.05]</sub> | **0.28**<br><sub>[0.03]</sub> | **0.16**<br><sub>[0.04]</sub> | **0.58**<br><sub>[0.03]</sub> | **0.21**<br><sub>[0.02]</sub> |
| TDOST | — | *0.63*<br><sub>[0.04]</sub> | 0.18<br><sub>[0.03]</sub> | 0.78<br><sub>[0.04]</sub> | 0.22<br><sub>[0.06]</sub> | *0.26*<br><sub>[0.05]</sub> | *0.15*<br><sub>[0.04]</sub> | *0.55*<br><sub>[0.03]</sub> | *0.18*<br><sub>[0.03]</sub> |

**budget = 250 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.72**<br><sub>[0.02]</sub> | **0.28**<br><sub>[0.05]</sub> | **0.86**<br><sub>[0.02]</sub> | **0.40**<br><sub>[0.06]</sub> | **0.30**<br><sub>[0.03]</sub> | **0.18**<br><sub>[0.06]</sub> | **0.63**<br><sub>[0.01]</sub> | **0.29**<br><sub>[0.03]</sub> |
| TDOST | — | 0.67<br><sub>[0.03]</sub> | 0.22<br><sub>[0.05]</sub> | *0.83*<br><sub>[0.02]</sub> | 0.30<br><sub>[0.06]</sub> | **0.31**<br><sub>[0.04]</sub> | *0.17*<br><sub>[0.05]</sub> | *0.60*<br><sub>[0.02]</sub> | 0.23<br><sub>[0.03]</sub> |

**all training windows**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.81**<br><sub>[0.00]</sub> | **0.51**<br><sub>[0.01]</sub> | **0.90**<br><sub>[0.00]</sub> | **0.47**<br><sub>[0.07]</sub> | 0.41<br><sub>[0.01]</sub> | 0.40<br><sub>[0.01]</sub> | 0.70<br><sub>[0.00]</sub> | 0.46<br><sub>[0.02]</sub> |
| TDOST | — | *0.78*<br><sub>[0.01]</sub> | 0.45<br><sub>[0.02]</sub> | *0.89*<br><sub>[0.01]</sub> | *0.46*<br><sub>[0.01]</sub> | **0.87**<br><sub>[0.01]</sub> | **0.61**<br><sub>[0.02]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.51**<br><sub>[0.01]</sub> |


### Baselines, L2 all classes

**budget = 100 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.70**<br><sub>[0.05]</sub> | **0.27**<br><sub>[0.04]</sub> | **0.82**<br><sub>[0.02]</sub> | **0.35**<br><sub>[0.07]</sub> | *0.71*<br><sub>[0.04]</sub> | *0.24*<br><sub>[0.05]</sub> | **0.75**<br><sub>[0.02]</sub> | **0.29**<br><sub>[0.04]</sub> |
| TDOST | — | **0.70**<br><sub>[0.03]</sub> | *0.25*<br><sub>[0.03]</sub> | 0.79<br><sub>[0.04]</sub> | 0.27<br><sub>[0.06]</sub> | **0.73**<br><sub>[0.02]</sub> | **0.25**<br><sub>[0.02]</sub> | *0.74*<br><sub>[0.02]</sub> | 0.26<br><sub>[0.03]</sub> |

**budget = 250 activity labels**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.74**<br><sub>[0.02]</sub> | **0.29**<br><sub>[0.03]</sub> | **0.86**<br><sub>[0.03]</sub> | **0.43**<br><sub>[0.08]</sub> | *0.74*<br><sub>[0.04]</sub> | *0.27*<br><sub>[0.05]</sub> | **0.78**<br><sub>[0.01]</sub> | **0.33**<br><sub>[0.03]</sub> |
| TDOST | — | *0.72*<br><sub>[0.01]</sub> | *0.28*<br><sub>[0.03]</sub> | 0.83<br><sub>[0.03]</sub> | 0.34<br><sub>[0.07]</sub> | **0.75**<br><sub>[0.01]</sub> | **0.28**<br><sub>[0.02]</sub> | *0.77*<br><sub>[0.01]</sub> | *0.30*<br><sub>[0.03]</sub> |

**all training windows**

| Method | Representation | milan W | milan M | aruba W | aruba M | cairo W | cairo M | average W | average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepCASAS | — | **0.81**<br><sub>[0.00]</sub> | **0.48**<br><sub>[0.00]</sub> | **0.91**<br><sub>[0.00]</sub> | **0.57**<br><sub>[0.09]</sub> | *0.82*<br><sub>[0.01]</sub> | **0.45**<br><sub>[0.03]</sub> | **0.85**<br><sub>[0.00]</sub> | **0.50**<br><sub>[0.03]</sub> |
| TDOST | — | *0.79*<br><sub>[0.01]</sub> | 0.43<br><sub>[0.03]</sub> | *0.90*<br><sub>[0.00]</sub> | 0.51<br><sub>[0.02]</sub> | **0.83**<br><sub>[0.17]</sub> | **0.44**<br><sub>[0.18]</sub> | *0.84*<br><sub>[0.06]</sub> | 0.46<br><sub>[0.07]</sub> |

