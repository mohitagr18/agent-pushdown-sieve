# Benchmark Summary Tables

Generated: 2026-10-15 01:33:43 UTC

T_token baseline = 350 ms (fixed, trace-driven replay). ATTI = T_tool/(350+T_tool).

## Experiment 1 — Scale sweep: mean T_tool (ms)

|                               |    B1 |    B2 |    B3 |    B4 |   Proposed |
|:------------------------------|------:|------:|------:|------:|-----------:|
| ('100k', 'isolation_forest')  | 180.2 | 177.5 | 143.1 | 155.2 |      149.8 |
| ('100k', 'kmeans_clustering') | 193.5 | 215.4 | 151.8 | 143.8 |      140.3 |
| ('100k', 'linear_regression') |  35.1 |  29.7 |   9   |  20.7 |       11.5 |
| ('10k', 'isolation_forest')   |  93.5 |  82   |  82.8 |  78.7 |       76   |
| ('10k', 'kmeans_clustering')  |  18.1 |  20.4 |   9.6 |  16.7 |       13.2 |
| ('10k', 'linear_regression')  |  13.1 |  11.7 |   5.2 |  10.2 |        6.3 |
| ('1M', 'isolation_forest')    | 758.6 | 682.6 | 487.1 | 514.6 |      500.8 |
| ('1M', 'kmeans_clustering')   | 554.7 | 530.9 | 298.9 | 378.7 |      330.6 |
| ('1M', 'linear_regression')   | 261.2 | 215.8 |  42.2 |  69.1 |       56   |

## Experiment 1 — ATTI (mean)

|                               |    B1 |    B2 |    B3 |    B4 |   Proposed |
|:------------------------------|------:|------:|------:|------:|-----------:|
| ('100k', 'isolation_forest')  | 0.331 | 0.326 | 0.281 | 0.298 |      0.29  |
| ('100k', 'kmeans_clustering') | 0.272 | 0.286 | 0.216 | 0.221 |      0.21  |
| ('100k', 'linear_regression') | 0.091 | 0.078 | 0.025 | 0.055 |      0.032 |
| ('10k', 'isolation_forest')   | 0.206 | 0.189 | 0.19  | 0.183 |      0.177 |
| ('10k', 'kmeans_clustering')  | 0.049 | 0.054 | 0.027 | 0.045 |      0.036 |
| ('10k', 'linear_regression')  | 0.036 | 0.032 | 0.015 | 0.028 |      0.018 |
| ('1M', 'isolation_forest')    | 0.664 | 0.637 | 0.531 | 0.545 |      0.54  |
| ('1M', 'kmeans_clustering')   | 0.596 | 0.586 | 0.419 | 0.479 |      0.452 |
| ('1M', 'linear_regression')   | 0.416 | 0.379 | 0.104 | 0.16  |      0.135 |

## Experiment 1 — sigma_ship (mean)

|                               |    B1 |    B2 |   B3 |   B4 |   Proposed |
|:------------------------------|------:|------:|-----:|-----:|-----------:|
| ('100k', 'isolation_forest')  | 156.3 | 156.3 |  1   |  1.3 |        1.2 |
| ('100k', 'kmeans_clustering') | 156.3 | 156.3 |  1.1 |  1.4 |        1.3 |
| ('100k', 'linear_regression') | 234.4 | 234.4 |  1.4 |  1.8 |        1.7 |
| ('10k', 'isolation_forest')   | 447.7 | 447.7 |  1   |  1.3 |        1.2 |
| ('10k', 'kmeans_clustering')  | 447.7 | 447.7 |  1.1 |  1.4 |        1.3 |
| ('10k', 'linear_regression')  | 671.5 | 671.5 |  1.4 |  1.8 |        1.7 |
| ('1M', 'isolation_forest')    | 216   | 216   |  1.1 |  1.3 |        1.2 |
| ('1M', 'kmeans_clustering')   | 216   | 216   |  1.2 |  1.4 |        1.3 |
| ('1M', 'linear_regression')   | 324   | 324   |  1.5 |  1.8 |        1.7 |

## Experiment 1 — peak RSS MB (mean)

|                               |   B1 |   B2 |   B3 |   B4 |   Proposed |
|:------------------------------|-----:|-----:|-----:|-----:|-----------:|
| ('100k', 'isolation_forest')  |  399 |  399 |  400 |  369 |        361 |
| ('100k', 'kmeans_clustering') |  414 |  416 |  415 |  390 |        375 |
| ('100k', 'linear_regression') |  413 |  412 |  415 |  380 |        372 |
| ('10k', 'isolation_forest')   |  311 |  311 |  315 |  316 |        317 |
| ('10k', 'kmeans_clustering')  |  326 |  326 |  327 |  326 |        326 |
| ('10k', 'linear_regression')  |  324 |  323 |  323 |  323 |        323 |
| ('1M', 'isolation_forest')    |  740 |  753 |  660 |  440 |        400 |
| ('1M', 'kmeans_clustering')   |  761 |  771 |  676 |  452 |        411 |
| ('1M', 'linear_regression')   |  768 |  790 |  726 |  468 |        413 |

## Experiment 2 — Multi-turn IMG ablation, 1M rows: mean T_tool (ms)

| query_id      |    B1 |    B2 |    B3 |   Proposed_IMG |   Proposed_noIMG |
|:--------------|------:|------:|------:|---------------:|-----------------:|
| session_01_t1 | 518.1 | 315.1 | 217.6 |          269.1 |            263.3 |
| session_01_t2 | 991.6 | 819.3 | 809.7 |          836.9 |            830.8 |
| session_01_t3 | 552   | 427.8 | 513.9 |          423.6 |            473.4 |
| session_02_t1 | 401   | 117.6 |  67.2 |          101.8 |            108.1 |
| session_02_t2 | 464.4 | 244.8 | 248.5 |          237.8 |            267.3 |
| session_02_t3 | 530   | 249.3 | 194.2 |          258.4 |            454.4 |
| session_03_t1 | 269.7 |  77   |  37.1 |           53.4 |             51.1 |
| session_03_t2 | 367   | 184.4 | 181.3 |          178.8 |            200.3 |
| session_03_t3 | 207.5 |  36.6 |  32.4 |           27.3 |             62.5 |
| session_04_t1 | 458   | 240.8 | 131.4 |          196.6 |            180.6 |
| session_04_t2 | 834   | 630.3 | 599.4 |          627.3 |            646.8 |
| session_04_t3 | 745.8 | 461.5 | 504.6 |          514.7 |            563.5 |
| session_05_t1 | 417.9 | 126.2 |  78.7 |          133.7 |            131.5 |
| session_05_t2 | 502.5 | 336.4 | 323.1 |          306.3 |            311.3 |
| session_05_t3 | 478.4 | 507.9 | 427.7 |          317.9 |            310.9 |

## Experiment 2 — sigma_ship per turn (mean)

| query_id      |    B1 |    B2 |   B3 |   Proposed_IMG |   Proposed_noIMG |
|:--------------|------:|------:|-----:|---------------:|-----------------:|
| session_01_t1 |  20.5 |  20.5 |  2.5 |            3.4 |              3.4 |
| session_01_t2 |  48.1 |  48.1 |  2   |            2.2 |              2.2 |
| session_01_t3 |  67   |  67   |  2   |            2.2 |              2.2 |
| session_02_t1 |  80.7 |  80.7 |  2.5 |            3.2 |              3.2 |
| session_02_t2 | 308.4 | 308.4 |  1.9 |            2.1 |              2.1 |
| session_02_t3 | 438.7 | 438.7 |  1.9 |            2.1 |              2.1 |
| session_03_t1 | 211   | 211   |  2.5 |            3.3 |              3.3 |
| session_03_t2 | 500.3 | 500.3 |  1.7 |            2.2 |              2.2 |
| session_03_t3 | 730.8 | 730.8 |  1.7 |            2.2 |              2.2 |
| session_04_t1 |  34.6 |  34.6 |  3.1 |            3.4 |              3.4 |
| session_04_t2 |  68.9 |  68.9 |  2   |            2.2 |              2.2 |
| session_04_t3 |  94.8 |  94.8 |  2   |            2.2 |              2.2 |
| session_05_t1 |  58.7 |  58.7 |  2.5 |            3.2 |              3.2 |
| session_05_t2 | 221.3 | 221.3 |  1.9 |            2.1 |              2.1 |
| session_05_t3 | 313.4 | 313.4 |  1.9 |            2.1 |              2.1 |

## Experiment 3 — Resolver resilience

Healed 10/10 dirty/adversarial traces (aliases + misspelled categoricals). Fail-closed on hallucinated columns (tested).

Vocabulary scaling — resolution latency (ms):

- vocab_10: 0.012 ms
- vocab_100: 0.069 ms
- vocab_1000: 0.948 ms
- vocab_10000: 0.215 ms
- vocab_100000: 0.966 ms

## Numerical parity

Max |Δ| vs B1 reference across B3/B4/Proposed, all scales and kernels: 0.00e+00 (eps 1e-5). PASSED.
