# Dataset notes — dset_allspec_150_chunks_clean.nc

Inspected directly with netCDF4 (no assumptions). 29.5 GB, 70,637 events.

## Variables
| var | shape | dtype | meaning |
|---|---|---|---|
| chunk | (70637, 3, 2048) | float32 | raw 3-axis vibration waveform |
| image | (70637, 128, 128, 3) | int64 | precomputed spectrogram-style image |
| class | (70637,) | int64 | species label (raw ids: 0,1,2,3,4,5,6,7,8,9,11) |
| distance | (70637,) | float32 | sensor-source distance, meters, range 11.4-149.8 |
| seis | (70637,) | int64 | seismometer station id, 10 unique stations |

## Species map (from prior notebooks — verify against original data source before publishing)
0 Human, 1 Elephant, 2 Giraffe, 3 Lion, 4 Buffalo, 5 Zebra, 6 Wildebeest, 7 Hippo, 8 Cheetah,
9 Hyena, 11 unmapped/unknown (4 samples only).

## Class counts (severe long-tail — drives the Core-6 / Full-11 split strategy in PLAN.md §1.2)
0: 2073, 1: 62493, 2: 164, 3: 2, 4: 271, 5: 769, 6: 4321, 7: 42, 8: 10, 9: 488, 11: 4

## Open questions to resolve before finalizing splits
- Does the same physical footstep event appear across multiple `seis` stations? If yes, a random
  split leaks; need a group-aware split by event id (no explicit event-id column found — check
  whether grouping can be reconstructed, or ask the data source).
- What does raw class `11` represent? Not in the `SPECIES_NAMES` dict used in prior notebooks.
