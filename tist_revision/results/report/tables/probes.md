### L1 (per-house activities, resident id removed), rare classes masked

**100 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.75* | 0.47 | *0.86* | *0.60* | 0.34 | 0.26 | 0.65 | 0.44 |
| Linear probe | Sensor+ToD hist. | **0.76** | **0.51** | *0.86* | *0.60* | **0.77** | **0.52** | **0.80** | **0.54** |
| | | | | | | | | | |
| Linear probe | MLM | *0.74* | 0.46 | **0.87** | **0.61** | 0.35 | 0.28 | 0.65 | 0.45 |
| Linear probe | MLM-time | **0.76** | *0.49* | **0.87** | *0.60* | 0.48 | 0.36 | 0.70 | 0.48 |
| Linear probe | MLM + ToD | *0.75* | *0.48* | **0.87** | **0.61** | 0.68 | 0.44 | 0.77 | 0.51 |
| Linear probe | MLM-time + ToD | **0.76** | **0.50** | **0.87** | *0.60* | 0.65 | 0.45 | 0.76 | *0.52* |

**500 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.80* | *0.59* | *0.87* | *0.62* | 0.36 | 0.35 | 0.68 | 0.52 |
| Linear probe | Sensor+ToD hist. | **0.80** | **0.61** | **0.87** | *0.63* | **0.82** | **0.64** | **0.83** | **0.63** |
| | | | | | | | | | |
| Linear probe | MLM | *0.78* | 0.56 | **0.88** | *0.63* | 0.38 | 0.37 | 0.68 | 0.52 |
| Linear probe | MLM-time | *0.80* | *0.59* | **0.88** | **0.64** | 0.70 | 0.55 | 0.79 | 0.60 |
| Linear probe | MLM + ToD | **0.80** | *0.59* | **0.88** | *0.63* | *0.81* | *0.64* | **0.83** | *0.62* |
| Linear probe | MLM-time + ToD | **0.80** | **0.60** | **0.88** | **0.64** | *0.81* | **0.64** | **0.83** | **0.63** |

**all labelled windows (max 20k)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.82* | 0.63 | *0.90* | *0.66* | 0.39 | 0.39 | 0.70 | 0.56 |
| Linear probe | Sensor+ToD hist. | **0.84** | **0.67** | **0.90** | *0.67* | *0.84* | 0.69 | **0.86** | *0.68* |
| | | | | | | | | | |
| Linear probe | MLM | *0.83* | *0.64* | **0.90** | *0.67* | 0.43 | 0.44 | 0.72 | 0.59 |
| Linear probe | MLM-time | *0.83* | *0.64* | **0.90** | **0.69** | *0.84* | *0.70* | *0.86* | *0.68* |
| Linear probe | MLM + ToD | **0.84** | **0.67** | **0.90** | *0.67* | **0.85** | **0.72** | **0.86** | **0.69** |
| Linear probe | MLM-time + ToD | *0.83* | *0.64* | **0.90** | **0.69** | **0.85** | *0.71* | **0.86** | *0.68* |


### L1, all classes

**100 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.74* | *0.35* | *0.86* | *0.42* | 0.33 | 0.24 | 0.65 | 0.34 |
| Linear probe | Sensor+ToD hist. | **0.75** | **0.37** | *0.86* | *0.42* | **0.77** | **0.48** | **0.79** | **0.42** |
| | | | | | | | | | |
| Linear probe | MLM | *0.73* | 0.34 | **0.87** | **0.43** | 0.35 | 0.25 | 0.65 | 0.34 |
| Linear probe | MLM-time | **0.75** | *0.37* | **0.87** | *0.42* | 0.47 | 0.32 | 0.70 | 0.37 |
| Linear probe | MLM + ToD | *0.74* | *0.36* | **0.87** | **0.43** | 0.68 | 0.40 | 0.76 | *0.40* |
| Linear probe | MLM-time + ToD | **0.76** | **0.37** | **0.87** | *0.42* | 0.65 | 0.41 | 0.76 | *0.40* |

**500 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.79* | *0.47* | *0.87* | *0.47* | 0.36 | 0.36 | 0.67 | 0.43 |
| Linear probe | Sensor+ToD hist. | **0.80** | *0.49* | **0.87** | *0.48* | **0.82** | *0.61* | **0.83** | *0.52* |
| | | | | | | | | | |
| Linear probe | MLM | *0.78* | 0.45 | **0.88** | *0.47* | 0.38 | 0.37 | 0.68 | 0.43 |
| Linear probe | MLM-time | *0.79* | *0.47* | **0.88** | **0.48** | 0.69 | 0.54 | 0.79 | 0.50 |
| Linear probe | MLM + ToD | **0.80** | *0.48* | **0.88** | *0.48* | *0.81* | *0.61* | **0.83** | *0.52* |
| Linear probe | MLM-time + ToD | **0.80** | **0.49** | **0.88** | **0.48** | *0.81* | **0.62** | **0.83** | **0.53** |

**all labelled windows (max 20k)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.82* | *0.54* | *0.89* | 0.67 | 0.39 | 0.41 | 0.70 | 0.54 |
| Linear probe | Sensor+ToD hist. | *0.83* | 0.53 | **0.90** | 0.69 | *0.84* | 0.68 | **0.86** | 0.63 |
| | | | | | | | | | |
| Linear probe | MLM | *0.83* | **0.57** | **0.90** | 0.69 | 0.43 | 0.46 | 0.72 | 0.58 |
| Linear probe | MLM-time | *0.82* | *0.56* | **0.90** | *0.73* | *0.84* | *0.70* | *0.86* | **0.66** |
| Linear probe | MLM + ToD | **0.84** | **0.57** | **0.90** | *0.71* | **0.85** | **0.71** | **0.86** | **0.66** |
| Linear probe | MLM-time + ToD | *0.82* | *0.55* | **0.90** | **0.73** | **0.85** | *0.71* | **0.86** | **0.67** |


### L2 (TDOST merged classes), all classes --- comparable to TDOST / DeepCASAS

**100 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.76* | 0.36 | *0.87* | *0.47* | 0.73 | 0.31 | 0.78 | 0.38 |
| Linear probe | Sensor+ToD hist. | **0.77** | **0.40** | *0.87* | *0.47* | **0.83** | **0.43** | **0.82** | **0.43** |
| | | | | | | | | | |
| Linear probe | MLM | *0.74* | 0.36 | **0.88** | **0.49** | 0.75 | 0.33 | 0.79 | 0.39 |
| Linear probe | MLM-time | *0.76* | *0.38* | **0.88** | *0.48* | 0.79 | 0.37 | *0.81* | *0.41* |
| Linear probe | MLM + ToD | *0.75* | *0.38* | **0.88** | **0.49** | 0.79 | 0.38 | *0.81* | *0.41* |
| Linear probe | MLM-time + ToD | *0.76* | *0.39* | **0.88** | *0.48* | *0.81* | 0.40 | **0.82** | *0.42* |

**500 random labelled windows**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.80* | 0.46 | *0.88* | *0.52* | 0.80 | 0.42 | 0.83 | 0.47 |
| Linear probe | Sensor+ToD hist. | **0.81** | **0.50** | **0.88** | **0.53** | **0.88** | **0.52** | **0.86** | **0.52** |
| | | | | | | | | | |
| Linear probe | MLM | *0.79* | 0.44 | **0.88** | *0.52* | 0.80 | 0.44 | 0.82 | 0.47 |
| Linear probe | MLM-time | *0.79* | 0.46 | **0.88** | *0.52* | 0.85 | 0.48 | *0.84* | *0.49* |
| Linear probe | MLM + ToD | *0.80* | *0.47* | **0.88** | **0.53** | *0.87* | *0.52* | *0.85* | *0.51* |
| Linear probe | MLM-time + ToD | *0.80* | *0.48* | **0.88** | *0.52* | *0.87* | *0.52* | *0.85* | *0.51* |

**all labelled windows (max 20k)**

| Method | Representation | Milan W | Milan M | Aruba W | Aruba M | Cairo W | Cairo M | Average W | Average M |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | Sensor hist. | *0.82* | 0.49 | *0.90* | 0.77 | 0.82 | 0.46 | 0.85 | 0.57 |
| Linear probe | Sensor+ToD hist. | **0.84** | *0.56* | *0.90* | 0.78 | **0.91** | 0.59 | **0.88** | 0.65 |
| | | | | | | | | | |
| Linear probe | MLM | *0.83* | 0.55 | **0.91** | 0.79 | 0.83 | 0.52 | *0.86* | 0.62 |
| Linear probe | MLM-time | *0.83* | *0.57* | **0.91** | **0.83** | *0.90* | *0.65* | *0.88* | **0.68** |
| Linear probe | MLM + ToD | **0.84** | **0.59** | **0.91** | 0.79 | **0.91** | **0.65** | **0.89** | *0.68* |
| Linear probe | MLM-time + ToD | *0.83* | *0.57* | **0.91** | **0.83** | **0.91** | **0.66** | **0.88** | **0.68** |

