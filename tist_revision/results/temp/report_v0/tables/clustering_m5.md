### L1 (per-house activities, resident id removed), rare classes masked (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.74** | **0.50** | **0.87** | **0.63** | 0.28 | 0.29 | 0.63 | **0.47** |
| k-means | Sensor+ToD hist. | *0.71* | 0.35 | 0.79 | 0.37 | **0.78** | **0.52** | **0.76** | 0.41 |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.75** | **0.55** | **0.87** | **0.61** | 0.30 | 0.28 | 0.64 | 0.48 |
| k-means | Sensor+ToD hist. | *0.72* | 0.42 | *0.86* | 0.57 | **0.82** | **0.62** | **0.80** | **0.54** |


### L1, all classes (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.73** | **0.37** | **0.87** | **0.44** | 0.28 | 0.26 | 0.63 | **0.36** |
| k-means | Sensor+ToD hist. | *0.71* | 0.26 | 0.79 | 0.26 | **0.78** | **0.47** | **0.76** | *0.33* |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.74** | **0.40** | **0.87** | **0.43** | 0.30 | 0.26 | 0.64 | 0.36 |
| k-means | Sensor+ToD hist. | *0.71* | 0.31 | *0.85* | *0.41* | **0.82** | **0.56** | **0.80** | **0.43** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS (m = 5)

**$K = 20$ clusters (100 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.76** | **0.42** | **0.87** | **0.48** | 0.74 | 0.30 | **0.79** | **0.40** |
| k-means | Sensor+ToD hist. | 0.72 | 0.30 | 0.79 | 0.31 | **0.84** | **0.40** | *0.78* | 0.34 |

**$K = 50$ clusters (250 labels)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.75** | **0.41** | **0.88** | **0.48** | 0.75 | 0.30 | 0.79 | 0.40 |
| k-means | Sensor+ToD hist. | *0.74* | 0.37 | *0.86* | **0.47** | **0.87** | **0.43** | **0.82** | **0.43** |

