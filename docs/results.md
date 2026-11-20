# Results

Metrics are mean ± s.d. over test volumes (isolated: Dice, panoptic quality; contiguous: Dice, centreline Dice).

## Final test

| Entry | isolated Dice | isolated PQ | contiguous Dice | contiguous clDice |
|---|---|---|---|---|
| Routed + anchored (Task 1) | 0.5474 ± 0.2620 | 0.3604 ± 0.2424 | 0.6529 ± 0.1550 | 0.7380 ± 0.1522 |
| v15 (reference) | 0.5474 ± 0.2620 | 0.3604 ± 0.2424 | 0.6528 ± 0.1547 | 0.7372 ± 0.1529 |

## Preliminary test (26 volumes)

| Entry | isolated Dice | isolated PQ | contiguous Dice | contiguous clDice | mean |
|---|---|---|---|---|---|
| v15 (reference) | 0.6287 ± 0.1604 | 0.4534 ± 0.1873 | 0.6332 ± 0.1296 | 0.7473 ± 0.0721 | 0.616 |
| Routed + anchored (Task 1) | 0.6287 ± 0.1604 | 0.4534 ± 0.1873 | 0.6258 ± 0.1283 | 0.7300 ± 0.0740 | 0.610 |

Development history on the preliminary test: v11 (percentile normalisation, AdamW) 0.537; v15 (recipe E + SSL) 0.616;
v17 (geometry-completion pretraining, slowed encoder, 112³) 0.573.

## Held-out split (9 volumes of seen types, v15a never trained on them)

| Model | isolated Dice | isolated PQ | contiguous Dice | contiguous clDice | mean |
|---|---|---|---|---|---|
| v11 | 0.651 | 0.325 | 0.733 | 0.849 | 0.640 |
| v15a | 0.710 | 0.378 | 0.739 | 0.804 | 0.658 |

Contiguous held-out volumes, v15a + a 128³ expert trained without them:

| Contiguous route | Dice | clDice |
|---|---|---|
| v15a alone, threshold 0.45 | 0.753 | 0.814 |
| 50/50 blend, anchored threshold (submitted rule) | 0.752 | 0.816 |
