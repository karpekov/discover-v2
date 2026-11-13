### L1 (per-house activities, resident id removed), rare classes masked (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.53** | **0.37** | **0.78** | **0.47** | 0.11 | 0.07 | **0.47** | **0.30** |
| k-means | Sensor+ToD hist. | 0.43 | 0.27 | 0.45 | 0.15 | **0.43** | **0.25** | 0.44 | 0.22 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.65** | **0.37** | **0.69** | **0.34** | 0.13 | 0.08 | **0.49** | 0.26 |
| k-means | Sensor+ToD hist. | 0.55 | *0.36* | 0.45 | 0.30 | **0.34** | **0.23** | 0.45 | **0.30** |


### L1, all classes (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.53** | **0.27** | **0.78** | **0.33** | 0.11 | 0.06 | **0.47** | **0.22** |
| k-means | Sensor+ToD hist. | 0.42 | 0.20 | 0.45 | 0.10 | **0.42** | **0.22** | 0.43 | 0.17 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.65** | **0.33** | **0.68** | **0.24** | 0.13 | 0.07 | **0.49** | *0.21* |
| k-means | Sensor+ToD hist. | 0.55 | 0.27 | 0.45 | *0.21* | **0.34** | **0.21** | 0.45 | **0.23** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.54** | **0.27** | **0.78** | **0.41** | 0.36 | 0.12 | **0.56** | **0.27** |
| k-means | Sensor+ToD hist. | 0.44 | 0.21 | 0.45 | 0.13 | **0.48** | **0.22** | 0.46 | 0.19 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.65** | **0.29** | **0.68** | **0.30** | 0.35 | 0.11 | **0.56** | **0.23** |
| k-means | Sensor+ToD hist. | 0.55 | 0.26 | 0.46 | 0.26 | **0.38** | **0.17** | 0.46 | **0.23** |

