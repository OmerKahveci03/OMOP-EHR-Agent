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

OMOP concepts were downloaded from [OHDSI Athena](https://athena.ohdsi.org/)
to `$HPG_BASE_DIR/vocabulary_download_v5/CONCEPT.csv`. Preprocessing combines
them with MIMIC's local concepts in `cleaned_data/concept.parquet`.
