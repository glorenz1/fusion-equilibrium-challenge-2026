# Training data and local evaluation splits

Use the official [Hugging Face dataset](https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge)
and [organizer data documentation](https://fusion-equilibrium-challenge.sophelio.io/data/).
The release is CC BY 4.0; retain the dataset attribution when distributing work.
The code in this repository is covered by its own `LICENSE`.

`config/dataset.json` pins repository `Sophelio/fusion-equilibrium-challenge` at
commit `1e280905b85f2a6fdde7e06fca8cf3a1edf447cb`. Verified on 2026-10-16.

| Config | Files | Contents | Used for |
| --- | ---: | --- | --- |
| `diii_d_train` | 7,041 | Inputs and equilibrium targets | Local training and held-shot evaluation |
| `diii_d_public_test` | 874 | Inputs, no targets | Predictions for organizer scoring |

MAST data is outside this FEC1 recipe. Download commands select only DIII-D.
The downloader uses the immutable revision and the repository's LFS content
hashes, checks counts and byte lengths, and atomically installs each verified
parquet. Partial downloads are removed; successfully completed files can be
reused on the next invocation. Budget disk space and network transfer before
starting the full download. `--workers` controls concurrent downloads.

```bash
.venv/bin/python -m fec1 download --output data --split diii_d_train
.venv/bin/python -m fec1 download --output data --split diii_d_public_test
```

Expected local layout:

```text
data/
  diii_d_train/
    d3d_shot_XXXXXXXXXX.parquet
    ...
    download.json
  diii_d_public_test/
    d3d_shot_XXXXXXXXXX.parquet
    ...
    download.json
```

Each parquet contains exactly one shot. Arrays are nested inside that row;
`efit_times` identifies the target frames and `efit_grid_R/Z` define the 65x65
grid. Preserve the original filenames. Public submission indexing follows
sorted filenames, matching the release's parquet file order.

Already-downloaded official parquet files can be used with `prepare --data
/path/to/diii_d_train`. Preparation requires the exact committed 7,041 identities
and checks the original frame counts. It hashes all raw file contents into the
run record. If `download.json` is present, it verifies those hashes against it;
without that manifest, the run records the local bytes and leaves the upstream
revision unspecified. Prefer the supplied downloader for verified provenance.

The included preprocessing applies the DIII-D plasma-current time-axis fix.
You do not need to clone the starter kit or preprocess the parquet files with
another script. The organizers' [data-fix announcement](https://fusion-equilibrium-challenge.sophelio.io/announcements/)
states that the released v1.1.0 data plus the input fix is the final training
release. Input columns for prediction are projected explicitly; target-derived
columns are excluded.

The local shot identities in `repro/splits/shots.json` are fixed:

| Local role | Shots | Use |
| --- | ---: | --- |
| Train | 5,677 | Nested shot-disjoint fitting and outer OOF reporting |
| Development | 689 | Evaluate the frozen five-member ensemble |
| Frozen audit | 675 | Excluded from fits, selection and default evaluation |

The preparation cache contains all released labelled shots. Supervision builders
read train shots only. Development targets are opened by evaluation only after
its predictions are frozen. Per-fold inner selection shots select epochs and
calibration; outer shots are excluded from that fold's carrier, scalar and field
fits. No public hidden labels are released or used by this workflow.

The `smoke` command uses separately generated synthetic data and a reduced,
isolated split. Its outputs are tagged `synthetic_smoke_only` and are never
reported as results on the challenge dataset.
