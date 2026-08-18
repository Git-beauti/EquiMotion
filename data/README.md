# Car-following Trajectory Dataset

This directory contains the standardized train/test split used for the
following-automated-vehicle longitudinal trajectory prediction task. The
dataset combines multi-source automated-driving and ACC car-following
trajectories in a common segment format.

| File | Rows | Segments | Size | SHA-256 |
|---|---:|---:|---:|---|
| `train.csv` | 1,355,100 | 4,517 | 248,199,591 B | `b05895ef1ce98e313cd7135d929130d33917c8cd22ce0008011a8623bd4bcf39` |
| `test.csv` | 581,100 | 1,937 | 64,946,701 B | `8648a36c7e01f112dd9965a63f016d97715b3986986d24b37604f3b448459a61` |

Both files use 300 rows per `Segment_ID` at a 0.1 s interval. The first 100
rows are the observed FAV history and the remaining 200 rows are the prediction
horizon. Future LV states remain visible in the test set; future FAV targets do
not.

Columns:

```text
Segment_ID, Time_Index, ID_LV, Type_LV, Pos_LV, Speed_LV, Acc_LV,
ID_FAV, Pos_FAV, Speed_FAV, Acc_FAV, Spatial_Gap, Spatial_Headway, Speed_Diff
```

The standardized files were distributed through the
[ET3 open trajectory benchmark](https://ieee-et3-challenge.com/). The CSV files
are tracked with Git LFS; run `git lfs pull` after cloning.
