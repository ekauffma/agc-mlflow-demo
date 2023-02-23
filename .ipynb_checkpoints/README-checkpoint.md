# agc-mlflow-demo
Demonstration of Training Pipeline for AGC Demo Day #2

This demo provides an example of how `mlflow` can be used during training in the AGC context. A simple signal/background classification is used here for ease of demonstration. Some physics details may not be completely realistic here since this is mainly meant to serve as a demonstration of the training mechanics.

More information on `mlflow` can be found [here](https://mlflow.org/).

Additionally, `xgboost` and `sklearn` are used here, but the approach can be modified to most popular machine learning libraries, including `tensorflow` and `pytorch`.

The hyperparameter optimization workflow structure is similar to the one described in this article: [https://medium.com/@chiefhustler/hyperparameter-tuning-with-dask-distributed-and-mlflow-ca6a4a275a2e](https://medium.com/@chiefhustler/hyperparameter-tuning-with-dask-distributed-and-mlflow-ca6a4a275a2e)