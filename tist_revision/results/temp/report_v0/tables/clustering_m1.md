### L1 (per-house activities, resident id removed), rare classes masked (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.65** | **0.42** | **0.87** | **0.64** | 0.27 | 0.30 | 0.60 | **0.46** |
| k-means | Sensor+ToD hist. | 0.61 | 0.35 | 0.73 | 0.35 | **0.75** | **0.45** | **0.70** | 0.39 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.74** | **0.45** | **0.87** | **0.59** | 0.28 | 0.27 | 0.63 | 0.44 |
| k-means | Sensor+ToD hist. | 0.66 | 0.40 | 0.81 | 0.56 | **0.66** | **0.48** | **0.71** | **0.48** |


### L1, all classes (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.65** | **0.31** | **0.87** | **0.49** | 0.27 | 0.27 | 0.60 | **0.36** |
| k-means | Sensor+ToD hist. | 0.60 | 0.26 | 0.73 | 0.24 | **0.75** | **0.41** | **0.70** | 0.30 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.74** | **0.38** | **0.86** | **0.46** | 0.28 | 0.24 | 0.63 | *0.36* |
| k-means | Sensor+ToD hist. | 0.66 | 0.30 | 0.81 | 0.40 | **0.66** | **0.44** | **0.71** | **0.38** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS (m = 1)

**$K = 20$ clusters (20 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.66** | **0.34** | **0.88** | **0.54** | 0.74 | 0.32 | **0.76** | **0.40** |
| k-means | Sensor+ToD hist. | 0.61 | 0.26 | 0.73 | 0.29 | **0.82** | **0.41** | 0.72 | 0.32 |

**$K = 50$ clusters (50 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.74** | **0.37** | **0.87** | **0.53** | 0.68 | 0.27 | **0.76** | **0.39** |
| k-means | Sensor+ToD hist. | 0.66 | 0.30 | 0.81 | 0.41 | **0.74** | **0.40** | *0.74* | *0.37* |

