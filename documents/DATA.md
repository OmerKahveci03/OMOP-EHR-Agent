Data used:

https://physionet.org/content/mimic-iv-demo-omop/0.9/

Downloaded to HiPerGator using:

```bash
# NOTE: You will need credentialled access and a password
wget -r -N -c -np https://physionet.org/files/mimic-iv-demo-omop/0.9/
```

The dataset contains 100 patients. Metadata of the dataset is saved under
`$HPG_BASE_DIR/cleaned_data/metadata/metadata_master.csv`. This file is created
by running `src/hipergator/preprocess.py`.
This file is one row per column, and it tells us the inferred dtype, file type it came from,
number of rows in the file type, and number of unique patients if the file type had a `person_id`
column.

The Athena vocabulary download must be extracted to
`$HPG_BASE_DIR/vocabulary_download_v5/CONCEPT.csv` before preprocessing. The script
streams this tab-delimited file and combines it with MIMIC's local
`2b_concept.csv` into `cleaned_data/concept.parquet`. Local MIMIC concepts take
precedence if an ID appears in both sources. The original CSV files are retained.
Run `python -m src.hipergator.preprocess` after the upload completes; use
`--overwrite` to rebuild all cleaned tables.
