### L1 (per-house activities, resident id removed), rare classes masked

**three-house average; clustering columns = k-means / SCAN / DEC, probe columns = same representation**

| Method | Representation | K20, m5 W | K20, m5 M | 100 labels W | 100 labels M | K50, m5 W | K50, m5 M | 250 labels W | 250 labels M | K50, m all W | K50, m all M | all labels W | all labels M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k-means / probe | Sensor hist. | 0.38 | *0.24* | 0.50 | *0.26* | 0.49 | *0.29* | 0.52 | 0.33 | *0.53* | **0.35** | 0.54 | 0.38 |
| k-means / probe | Sensor+ToD hist. | **0.54** | **0.24** | **0.55** | **0.29** | **0.54** | **0.32** | **0.60** | **0.40** | **0.56** | 0.31 | **0.64** | **0.47** |
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
| k-means / probe | Sensor hist. | 0.38 | *0.17* | 0.50 | *0.20* | 0.49 | 0.21 | 0.51 | 0.25 | *0.53* | **0.27** | 0.54 | 0.29 |
| k-means / probe | Sensor+ToD hist. | **0.53** | **0.19** | **0.55** | **0.23** | **0.54** | **0.25** | **0.60** | **0.32** | **0.56** | 0.23 | **0.64** | **0.36** |
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
| k-means / probe | Sensor hist. | 0.40 | **0.20** | *0.56* | *0.24* | 0.51 | 0.23 | 0.56 | 0.27 | *0.54* | **0.27** | 0.57 | 0.30 |
| k-means / probe | Sensor+ToD hist. | **0.55** | **0.20** | **0.58** | **0.25** | **0.57** | **0.28** | **0.62** | **0.32** | **0.56** | *0.26* | **0.66** | **0.37** |
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

