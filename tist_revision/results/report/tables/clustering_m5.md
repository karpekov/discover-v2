### L1 (per-house activities, resident id removed), rare classes masked (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.45 | **0.36** | *0.67* | **0.34** | 0.04 | 0.01 | 0.38 | *0.24* |
| k-means | Sensor+ToD hist. | **0.56** | 0.31 | **0.68** | 0.23 | **0.37** | **0.20** | **0.54** | **0.24** |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.70** | **0.47** | **0.69** | *0.34* | 0.09 | 0.06 | 0.49 | *0.29* |
| k-means | Sensor+ToD hist. | 0.51 | 0.32 | **0.69** | **0.37** | **0.43** | **0.27** | **0.54** | **0.32** |


### L1, all classes (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.45 | **0.27** | *0.66* | **0.24** | 0.04 | 0.01 | 0.38 | *0.17* |
| k-means | Sensor+ToD hist. | **0.56** | 0.23 | **0.68** | 0.16 | **0.36** | **0.18** | **0.53** | **0.19** |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.69** | **0.35** | **0.69** | *0.24* | 0.09 | 0.05 | 0.49 | 0.21 |
| k-means | Sensor+ToD hist. | 0.50 | 0.24 | **0.69** | **0.26** | **0.43** | **0.25** | **0.54** | **0.25** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | 0.48 | **0.29** | *0.66* | **0.30** | 0.06 | 0.01 | 0.40 | **0.20** |
| k-means | Sensor+ToD hist. | **0.56** | 0.25 | **0.68** | 0.20 | **0.41** | **0.15** | **0.55** | **0.20** |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.69** | **0.33** | **0.69** | *0.30* | 0.14 | 0.05 | 0.51 | 0.23 |
| k-means | Sensor+ToD hist. | 0.54 | 0.27 | **0.69** | **0.32** | **0.48** | **0.25** | **0.57** | **0.28** |

