# Dataset Example

The minimum dataset used by this project has this layout:

```text
minimum_dataset/
  metadata.xlsx
  A01/
    5.jpg
    7.jpg
    9.jpg
    11.jpg
    13.jpg
  A02/
    5.jpg
    7.jpg
    9.jpg
    11.jpg
    13.jpg
```

`metadata.xlsx` should contain one row per OCT volume/visit. The default image
folder column is `OCT_pathway`, and each value is a folder relative to the
dataset root.

Required columns for the bundled task configs:

```text
OCT_pathway
anti-VEGF injection
Diagnosis
IRF
SRF
SHRM
DRT
NV
PED
```

See `minimum_metadata_example.csv` for a small tabular example. Real OCT images
are not bundled in this public repository.
