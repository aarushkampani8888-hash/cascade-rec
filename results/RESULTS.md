# Results — ml-100k

943 users, 1682 items, 100000 interactions; 189 users in the simulated cold-start cohort (5 known interactions at tune time, 6 at test).

Leave-last-out, full-catalog ranking (no sampled negatives), test item untouched until the end.
95% CIs from 2,000 bootstrap resamples over users.

| Model | Recall@10 | NDCG@10 [95% CI] | NDCG@10 warm | NDCG@10 cold | Recall@20 | MRR | Coverage@10 |
|---|---|---|---|---|---|---|---|
| content | 0.0095 | 0.0044 [0.0015, 0.0080] | 0.0038 | 0.0069 | 0.0201 | 0.0077 | 0.369 |
| pop | 0.0817 | 0.0382 [0.0297, 0.0474] | 0.0387 | 0.0363 | 0.1177 | 0.0341 | 0.048 |
| itemknn | 0.1060 | 0.0595 [0.0479, 0.0718] | 0.0595 | 0.0594 | 0.1898 | 0.0602 | 0.351 |
| lightgcn | 0.1273 | 0.0650 [0.0533, 0.0781] | 0.0697 | 0.0460 | 0.2015 | 0.0616 | 0.498 |
| ease | 0.1389 | 0.0703 [0.0586, 0.0835] | 0.0741 | 0.0551 | 0.2121 | 0.0628 | 0.364 |
| sasrec | 0.1654 | 0.0898 [0.0759, 0.1040] | 0.0872 | 0.1002 | 0.2375 | 0.0813 | 0.404 |
| linear blend (tuned) | 0.1845 | 0.1026 [0.0876, 0.1184] | 0.1085 | 0.0790 | 0.2916 | 0.0946 | 0.414 |
| hybrid (learned ranker) | 0.2068 | 0.1120 [0.0964, 0.1283] | 0.1200 | 0.0800 | 0.2948 | 0.0993 | 0.455 |

## Is the hybrid actually better? (paired bootstrap, NDCG@10)

| vs. | hybrid − model (all) | 95% CI | P(diff ≤ 0) | hybrid − model (cold) | 95% CI |
|---|---|---|---|---|---|
| pop | +0.0738 | [+0.0573, +0.0905] | 0.000 | +0.0437 | [+0.0083, +0.0784] |
| itemknn | +0.0525 | [+0.0370, +0.0677] | 0.000 | +0.0206 | [-0.0075, +0.0502] |
| ease | +0.0417 | [+0.0288, +0.0544] | 0.000 | +0.0250 | [+0.0014, +0.0513] |
| content | +0.1076 | [+0.0923, +0.1233] | 0.000 | +0.0731 | [+0.0452, +0.1048] |
| lightgcn | +0.0470 | [+0.0327, +0.0606] | 0.000 | +0.0340 | [+0.0096, +0.0591] |
| sasrec | +0.0222 | [+0.0079, +0.0360] | 0.000 | -0.0202 | [-0.0517, +0.0084] |
| linear blend (tuned) | +0.0094 | [+0.0024, +0.0162] | 0.004 | +0.0011 | [-0.0117, +0.0151] |

## Retrieval stage

Union of retrievers gives 318 candidates/user; the test item is in the set for 85.4% of users (81.5% of cold users). That's the ceiling for the ranker.

| Retriever | top-N | recall@N |
|---|---|---|
| ease | 150 | 0.577 |
| sasrec | 150 | 0.674 |
| lightgcn | 150 | 0.644 |
| itemknn | 50 | 0.338 |
| content | 50 | 0.083 |
| pop | 25 | 0.138 |

## Ablations (drop one source's candidates and features, retrain ranker)

| Dropped | NDCG@10 | Δ vs full [95% CI] | NDCG@10 cold |
|---|---|---|---|
| pop | 0.1067 | -0.0052 [-0.0110, +0.0002] | 0.0815 |
| itemknn | 0.1042 | -0.0078 [-0.0145, -0.0014] | 0.0781 |
| ease | 0.1041 | -0.0079 [-0.0156, -0.0003] | 0.0809 |
| content | 0.1028 | -0.0092 [-0.0177, -0.0013] | 0.0820 |
| lightgcn | 0.1044 | -0.0076 [-0.0164, +0.0002] | 0.0785 |
| sasrec | 0.0828 | -0.0292 [-0.0416, -0.0174] | 0.0613 |
| side | 0.1076 | -0.0044 [-0.0101, +0.0012] | 0.0849 |

## What the ranker leans on (mean |SHAP| over each user's top-20)

| Feature group | warm users | cold users |
|---|---|---|
| sasrec | 1.074 | 1.067 |
| lightgcn | 0.677 | 0.713 |
| ease | 0.164 | 0.143 |
| content | 0.058 | 0.073 |
| taste | 0.012 | 0.014 |
| agreement | 0.010 | 0.010 |
| user | 0.010 | 0.011 |
| item | 0.009 | 0.011 |
| itemknn | 0.007 | 0.005 |
| pop | 0.007 | 0.007 |

Tree corrections only (what the boosted trees add on top of the linear blend):

| Feature group | warm users | cold users |
|---|---|---|
| lightgcn | 0.051 | 0.062 |
| sasrec | 0.032 | 0.029 |
| content | 0.013 | 0.016 |
| taste | 0.012 | 0.014 |
| agreement | 0.010 | 0.010 |
| user | 0.010 | 0.011 |
| item | 0.009 | 0.011 |
| itemknn | 0.007 | 0.005 |
| ease | 0.005 | 0.007 |
| pop | 0.004 | 0.004 |

## Example explanations

**User 259** (warm) — recently watched: Long Kiss Goodnight, The, French Kiss, Maltese Falcon, The, Dazed and Confused, Strange Days

1. **Star Wars** — often watched alongside things in your history; people whose overall taste resembles yours watched it
2. **Pulp Fiction** — people whose overall taste resembles yours watched it; matches genres you've been watching (Crime, Drama)
3. **Fargo** — people whose overall taste resembles yours watched it; often watched alongside things in your history
4. **Terminator 2: Judgment Day** — matches genres you've been watching (Action, Sci-Fi); often watched alongside things in your history
5. **Fugitive, The** — matches genres you've been watching (Action, Thriller); often watched alongside things in your history

**User 712** (cold-start) — recently watched: Sound of Music, The, Liar Liar, Room with a View, A, Roommates, Bedknobs and Broomsticks

1. **Pretty Woman** — follows on from what you've watched lately, especially Sound of Music, The; people whose overall taste resembles yours watched it
2. **Home Alone** — follows on from what you've watched lately, especially Sound of Music, The; matches genres you've been watching (Children's, Comedy)
3. **Sleepless in Seattle** — follows on from what you've watched lately, especially Sound of Music, The; people whose overall taste resembles yours watched it
4. **While You Were Sleeping** — people whose overall taste resembles yours watched it; matches genres you've been watching (Comedy, Romance)
5. **Grease** — people whose overall taste resembles yours watched it; follows on from what you've watched lately, especially Sound of Music, The

## Ranker selection (scored on the most recent training slice, never on test)

| leaves | min leaf | raw scores | start from blend | trees | val NDCG@10 |
|---|---|---|---|---|---|
| 7 | 400 | True | True | 46 | 0.1403 |
| 7 | 100 | True | True | 47 | 0.1400 |
| 31 | 400 | False | True | 15 | 0.1393 |
| 7 | 100 | False | True | 36 | 0.1390 |
| 7 | 400 | False | True | 34 | 0.1387 |
| 31 | 100 | True | True | 11 | 0.1386 |
| 31 | 100 | False | True | 12 | 0.1383 |
| 7 | 400 | True | False | 314 | 0.1381 |
| 31 | 400 | True | True | 100 | 0.1363 |
| 7 | 100 | False | False | 361 | 0.1352 |
| 31 | 100 | False | False | 63 | 0.1318 |
| 7 | 400 | False | False | 211 | 0.1317 |
| 7 | 100 | True | False | 156 | 0.1304 |
| 31 | 100 | True | False | 200 | 0.1302 |
| - | - | - | blend only | 0 | 0.1302 |
| 31 | 400 | False | False | 263 | 0.1296 |
| 31 | 400 | True | False | 155 | 0.1290 |

## Tuned settings

```
retrievers: {'ease_lambda': 50, 'lightgcn_epochs': 130, 'sasrec_epochs': 170}
ranker: {'num_leaves': 7, 'min_data_in_leaf': 400, 'raw_scores': True, 'init_blend': True} (46 trees)
blend weights: {'pop': 0.003, 'itemknn': 0.0, 'ease': 0.043, 'content': 0.07, 'lightgcn': 0.308, 'sasrec': 0.576}
```