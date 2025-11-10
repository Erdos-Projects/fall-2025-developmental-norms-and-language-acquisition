# LingPredict
Do Developmental Norms Predict L2 Difficulty? Modeling Duolingo Learners with WordBank Features

## Primary research question:
Do words acquired later in first-language development (L1) show higher error rates and slower learning in second language (L2) practice on Duolingo? We estimate the age at which a child learns a given word in L1 based on developmental norms as described by the [WordBank dataset](https://wordbank.stanford.edu/). We then compare this with the time it takes a [Duolingo](https://www.duolingo.com/) user to learn this word in L2 based on the [Duolingo SLAM Dataset](https://doi.org/10.7910/DVN/8SWHNO). See also the 2018 SLAM Task overview article [*Second Language Acquisition Modeling*](https://doi.org/10.18653/v1/W18-0506) by Settles, Brust, Gustafson, Hagiwara, and Madnani.

## Data:

### Duolingo SLAM Dataset:
-	Large corpus of data from over 6,000 Duolingo users, collected during their first 30 days of learning a language
-	Released in 2018 as part of the [Duolingo Shared Task on Second Language Acquisition Modeling (SLAM)](https://sharedtask.duolingo.com/2018.html).
-	Potential features to use: practice history, time since last exposure, item difficulty, etc.
- Publicly available via [Harvard Dataverse](https://doi.org/10.7910/DVN/8SWHNO) or linked from Duolingo Research.

### WordBank Dataset:
-	Open database of children’s vocabulary growth.
-	Aggregates data from the [MacArthur–Bates Communicative Development Inventories](https://mb-cdi.stanford.edu/) (CDIs).
- Potential features: Age of Acquistion (AoA),category (noun, verb, semantic field), production/comprehension probabilities.
- Publicly available at [wordbank.stanford.edu](https://wordbank.stanford.edu).

## Instructions:
1. Create & activate the environment:
  ```
  conda env create -f environment.yml
  conda activate lingpredict   # or the name defined in environment.yml
  ```
2. Download the [SLAM dataset from Dataverse](https://doi.org/10.7910/DVN/8SWHNO). (Their license precludes us from adding these files to the repository.) Unzip `dataverse_files-2018.zip` and the resulting tarball files `data_en_es.tar.gz`, `data_es_en.tar.gz`, and `data_fr_en.tar.gz`. Move the resulting folders `data_en_es`, `data_es_en`, and `data_fr_en` into `data/raw`. The resulting file structure should appear as follows:
  ```
  data/
  ├── processed/
  │   ├── language_translation_table.csv
  │   ├── wordbank_en_logistic_fits.csv
  │   └── wordbank_es_logistic_fits.csv
  └── raw/
      ├── data_en_es/
      │   ├── en_es.slam.20190204.dev
      │   ├── en_es.slam.20190204.dev.key
      │   ├── en_es.slam.20190204.test
      │   ├── en_es.slam.20190204.test.key
      │   ├── en_es.slam.20190204.train
      │   └── (Metadata files: CHANGELOG.md, CITING.md, LICENSE.md)
      ├── data_es_en/
      │   └── (Files structured similarly to data_en_es)
      ├── data_fr_en/
      │   └── (Files structured similarly to data_en_es)
      └── Wordbank/
          └── (Supplementary linguistic files)
  ```
3.	Run `run_all.sh`.
