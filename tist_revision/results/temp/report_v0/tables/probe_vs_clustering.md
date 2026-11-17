### L1 (per-house activities, resident id removed), rare classes masked

**three-house average; clustering columns = k-means / SCAN / DEC, probe columns = same representation**

| Method | Representation | K20, m5 W | K20, m5 M | 100 labels W | 100 labels M | K50, m5 W | K50, m5 M | 250 labels W | 250 labels M | K50, m all W | K50, m all M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means / probe | Sensor hist. | 0.63 | **0.47** | 0.61 | 0.33 | 0.64 | 0.48 | 0.64 | 0.43 | 0.68 | 0.51 | 0.69 | 0.54 |
| k-means / probe | Sensor+ToD hist. | **0.76** | 0.41 | **0.71** | **0.39** | **0.80** | **0.54** | **0.75** | **0.50** | **0.82** | **0.58** | **0.85** | **0.65** |
| | | | | | | | | | | | | | |
| k-means / probe | MLM | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| SCAN | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| SCAN | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| DEC | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| DEC | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |


### L1, all classes

**three-house average; clustering columns = k-means / SCAN / DEC, probe columns = same representation**

| Method | Representation | K20, m5 W | K20, m5 M | 100 labels W | 100 labels M | K50, m5 W | K50, m5 M | 250 labels W | 250 labels M | K50, m all W | K50, m all M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means / probe | Sensor hist. | 0.63 | **0.36** | 0.60 | 0.26 | 0.64 | 0.36 | 0.63 | 0.34 | 0.67 | 0.41 | 0.69 | 0.46 |
| k-means / probe | Sensor+ToD hist. | **0.76** | *0.33* | **0.71** | **0.31** | **0.80** | **0.43** | **0.75** | **0.41** | **0.82** | **0.46** | **0.85** | **0.56** |
| | | | | | | | | | | | | | |
| k-means / probe | MLM | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| SCAN | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| SCAN | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| DEC | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| DEC | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS

**three-house average; clustering columns = k-means / SCAN / DEC, probe columns = same representation**

| Method | Representation | K20, m5 W | K20, m5 M | 100 labels W | 100 labels M | K50, m5 W | K50, m5 M | 250 labels W | 250 labels M | K50, m all W | K50, m all M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means / probe | Sensor hist. | **0.79** | **0.40** | *0.75* | *0.31* | 0.79 | 0.40 | *0.78* | *0.38* | *0.82* | *0.44* | 0.84 | 0.49 |
| k-means / probe | Sensor+ToD hist. | *0.78* | 0.34 | **0.76** | **0.33** | **0.82** | **0.43** | **0.80** | **0.41** | **0.84** | **0.45** | **0.88** | **0.56** |
| | | | | | | | | | | | | | |
| k-means / probe | MLM | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| k-means / probe | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| SCAN | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| SCAN | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |
| | | | | | | | | | | | | | |
| DEC | MLM-time | – | – | – | – | – | – | – | – | – | – | – | – |
| DEC | MLM-time + ToD | – | – | – | – | – | – | – | – | – | – | – | – |

