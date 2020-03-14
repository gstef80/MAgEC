# MAgEC

### Paper
Title: *"MAgEC: Using Non-Homogeneous Ensemble Consensus for 
Predicting Drivers in Unexpected Mechanical Ventilation"*.
<br>
Paper presented at AMIA 2021 Virtual Informatics Summit. 
<br>
### Code Structure
```shell
.
├── LICENSE
├── README.md
├── __init__.py
├── environment.yml
├── notebooks
│   ├── MAgEC_SHAP_RBO.ipynb
│   ├── MIMIC_MV.ipynb
│   ├── PIMA_MAGEC_PRESENTATION.ipynb
│   ├── __init__.py
├── rboREADME.md
├── requirements.txt
└── src
    ├── __init__.py
    ├── data
    │   └── diabetes.csv
    ├── magec_sensitivity.py
    ├── magec_utils.py
    ├── mimic_queries.py
    ├── mimic_utils.py
    ├── pima_utils.py
    └── rbo.py
```
1. Use `conda env create -f environment.yml` to create a virtual 
environment with required libraries. 
2. Build MIMIC-III DB (see MIMIC-III Data section)
3. Run `MIMIC_MV.ipynb`
4. Run `MAgEC_SHAP_RBO.ipynb`
RBO implementation by Changyao Chen (read `rboREADME.md`)


If you dont' have access to MIMIC-III you can check out
`PIMA_MAGEC_PRESENTATION.ipynb` for an application of MAgEC 
using the PIMA dataset.

### MIMIC-III Data
For the MIMIC-III datasets, after being granted access 
(https://mimic.physionet.org/gettingstarted/dbsetup/), 
build the database using the instructions at 
https://github.com/MIT-LCP/mimic-code/tree/master/buildmimic. 
<br>
**Note**: "mv_users" table is created in notebooks/MIMIC_MV.ipyng) 
<br>
A snapshot of the tables created is below: 
<br>
**MIMIC-III Tables**
```
+----------+------------------------------+--------+----------+
| Schema   | Name                         | Type   | Owner    |
|----------+------------------------------+--------+----------|
| mimiciii | "mimic".mimiciii"."mv_users" | table  | postgres |
| mimiciii | admissions                   | table  | postgres |
| mimiciii | callout                      | table  | postgres |
| mimiciii | caregivers                   | table  | postgres |
| mimiciii | chartevents                  | table  | postgres |
| mimiciii | chartevents_1                | table  | postgres |
| mimiciii | chartevents_10               | table  | postgres |
| mimiciii | chartevents_11               | table  | postgres |
| mimiciii | chartevents_12               | table  | postgres |
| mimiciii | chartevents_13               | table  | postgres |
| mimiciii | chartevents_14               | table  | postgres |
| mimiciii | chartevents_15               | table  | postgres |
| mimiciii | chartevents_16               | table  | postgres |
| mimiciii | chartevents_17               | table  | postgres |
| mimiciii | chartevents_2                | table  | postgres |
| mimiciii | chartevents_3                | table  | postgres |
| mimiciii | chartevents_4                | table  | postgres |
| mimiciii | chartevents_5                | table  | postgres |
| mimiciii | chartevents_6                | table  | postgres |
| mimiciii | chartevents_7                | table  | postgres |
| mimiciii | chartevents_8                | table  | postgres |
| mimiciii | chartevents_9                | table  | postgres |
| mimiciii | cptevents                    | table  | postgres |
| mimiciii | d_cpt                        | table  | postgres |
| mimiciii | d_icd_diagnoses              | table  | postgres |
| mimiciii | d_icd_procedures             | table  | postgres |
| mimiciii | d_items                      | table  | postgres |
| mimiciii | d_labitems                   | table  | postgres |
| mimiciii | datetimeevents               | table  | postgres |
| mimiciii | diagnoses_icd                | table  | postgres |
| mimiciii | drgcodes                     | table  | postgres |
| mimiciii | icustays                     | table  | postgres |
| mimiciii | inputevents_cv               | table  | postgres |
| mimiciii | inputevents_mv               | table  | postgres |
| mimiciii | labevents                    | table  | postgres |
| mimiciii | microbiologyevents           | table  | postgres |
| mimiciii | mimic_users_study            | table  | postgres |
| mimiciii | mv_users                     | table  | postgres |
| mimiciii | noteevents                   | table  | postgres |
| mimiciii | outputevents                 | table  | postgres |
| mimiciii | patients                     | table  | postgres |
| mimiciii | prescriptions                | table  | postgres |
| mimiciii | procedureevents_mv           | table  | postgres |
| mimiciii | procedures_icd               | table  | postgres |
| mimiciii | services                     | table  | postgres |
| mimiciii | transfers                    | table  | postgres |
+----------+------------------------------+--------+----------+
```