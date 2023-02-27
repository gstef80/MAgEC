# Table1.5

Table 1.5 is a Python application that can generate a table that is adjunct to a typical Table 1 (association statistics). Table 1.5 goes beyond static association by analyzing the impact that a change in each single feature has to changes in the outcome.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Table1.5.

```bash
pip install table15
```

## Code Structure
```
table15
├── LICENSE
├── README.md
│       ├── __init__.py
│       ├── magec_sensitivity.py
│       ├── mimic_queries.py
│       ├── mimic_utils.py
│       ├── rbo.py
│       └── table15
│           ├── __init__.py
│           ├── __main__.py
│           ├── magec_utils.py
│           ├── pima_utils.py
│           ├── pipeline_utils.py
│           ├── runner.py
│           ├── utils
│           │   ├── __init__.py
│           │   ├── data_utils.py
│           │   ├── magec_utils.py
│           │   ├── model_utils.py
│           │   ├── pima_utils.py
│           │   └── pipeline_utils.py
│           └── viewer.py
├── pyproject.toml
├── src
│   ├── __init__.py
│   ├── data
│   │   ├── diabetes.csv
│   │   ├── healthcare-dataset-stroke-data.csv
│   │   └── linear_data.csv
│   ├── table15
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── configs
│   │   │   ├── data_configs
│   │   │   │   ├── pima_full.yaml
│   │   │   │   ├── pima_lite.yaml
│   │   │   │   └── stroke_full.yaml
│   │   │   ├── model_configs
│   │   │   │   ├── deep_models_configs
│   │   │   │   │   └── multi_layer_perceptron_1.yaml
│   │   │   │   ├── ensemble_configs
│   │   │   │   │   ├── random_forrest_cc_1.yaml
│   │   │   │   │   └── voting_classifier_1.yaml
│   │   │   │   ├── linear_model_configs
│   │   │   │   │   ├── lr_1.yaml
│   │   │   │   │   ├── lr_2.yaml
│   │   │   │   │   └── lr_cv_1.yaml
│   │   │   │   └── svm_configs
│   │   │   │       ├── linear_svm_cc_1.yaml
│   │   │   │       └── svm_1.yaml
│   │   │   └── pipeline_configs
│   │   │       ├── linear.yaml
│   │   │       ├── pima.yaml
│   │   │       ├── stroke.yaml
│   │   │       └── synth_data.yaml
│   │   ├── configs.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── deep_models.py
│   │   │   ├── ensemble_models.py
│   │   │   ├── linear_models.py
│   │   │   ├── model.py
│   │   │   ├── model_factory.py
│   │   │   ├── svm_models.py
│   │   │   └── test_linear_model.py
│   │   ├── perturbations
│   │   │   ├── __init__.py
│   │   │   ├── group_perturbation.py
│   │   │   ├── perturbation.py
│   │   │   └── z_perturbation.py
│   │   ├── runner.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-38.pyc
│   │       │   ├── data_tables.cpython-38.pyc
│   │       │   ├── data_utils.cpython-38.pyc
│   │       │   ├── magec_utils.cpython-38.pyc
│   │       │   ├── model_utils.cpython-38.pyc
│   │       │   ├── models_container.cpython-38.pyc
│   │       │   └── pipeline_utils.cpython-38.pyc
│   │       ├── data_tables.py
│   │       ├── magec_utils.py
│   │       ├── models_container.py
│   │       └── pipeline_utils.py
│   └── table15.egg-info
└── tests
    ├── configs
    │   └── t_configs.yaml
    ├── test_perturbations.py
    └── test_pipeline_utils.py
```

## Usage

```python
import table15
table15.runner.run(`file_path`)
```

## Support
Issues and support can be directed to @KaleRP

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Authors and acknowledgment
Author of this project is @KaleRP.
Special thanks to @gstef80 for creating the original project this application was forked from.
Another special thanks to @beaunorgeot for originally conceiving this project.


## License

[MIT](https://choosealicense.com/licenses/mit/)