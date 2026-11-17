### L1 (per-house activities, resident id removed), rare classes masked (m = all)

**$K = 20$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.79** | **0.58** | **0.86** | **0.59** | 0.32 | 0.31 | 0.66 | **0.49** |
| k-means | Sensor+ToD hist. | 0.72 | 0.39 | 0.79 | 0.37 | **0.77** | **0.49** | **0.76** | 0.41 |

**$K = 50$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.80** | **0.60** | **0.87** | **0.61** | 0.35 | 0.33 | 0.68 | 0.51 |
| k-means | Sensor+ToD hist. | *0.78* | 0.53 | *0.86* | *0.60* | **0.81** | **0.61** | **0.82** | **0.58** |


### L1, all classes (m = all)

**$K = 20$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.79** | **0.42** | **0.86** | **0.41** | 0.32 | 0.28 | 0.66 | **0.37** |
| k-means | Sensor+ToD hist. | 0.71 | 0.29 | 0.79 | 0.26 | **0.77** | **0.44** | **0.76** | 0.33 |

**$K = 50$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.80** | **0.49** | **0.87** | **0.43** | 0.35 | 0.30 | 0.67 | 0.41 |
| k-means | Sensor+ToD hist. | *0.78* | 0.39 | *0.86* | *0.42* | **0.80** | **0.56** | **0.82** | **0.46** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS (m = all)

**$K = 20$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.79** | **0.45** | **0.87** | **0.48** | 0.74 | 0.30 | **0.80** | **0.41** |
| k-means | Sensor+ToD hist. | 0.73 | 0.31 | 0.79 | 0.31 | **0.84** | **0.40** | *0.79* | 0.34 |

**$K = 50$ clusters**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means | Sensor hist. | **0.81** | **0.46** | **0.88** | **0.49** | 0.79 | 0.36 | *0.82* | *0.44* |
| k-means | Sensor+ToD hist. | *0.79* | 0.42 | *0.87* | *0.48* | **0.87** | **0.43** | **0.84** | **0.45** |

