# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *HyperOpt*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Pandas Dataframe that can be used passed to a model.

# COMMAND ----------

readPath = 'dbfs:/mnt/azureml/real_estate.csv'

df = (spark
      .read
      .option('header', True)
      .option("inferSchema",True)
      .format('csv')
      .load(readPath)
     ).toPandas().set_index('No')

# COMMAND ----------

label = 'Y house price of unit area'
X = df.drop(label, axis=1)
y = df[label]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing libraries

# COMMAND ----------

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ## Splitting dataset into test and train set

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importing routines for use in HyperOpt

# COMMAND ----------

import pandas as pd
import mlflow
import sklearn
import mlflow.spark
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the training model
# MAGIC Here we create a function that is used for training. This is customisable to have whatever you like. It takes parameters as arguments that are then logged using MLFlow. It outputs the loss function which can be coded to be whatever the user like. Here it is chosen to be the mean squared error. 

# COMMAND ----------

# define model + hyperparameters to optimise
def train_model(max_features, min_samples_split, min_samples_leaf):
    with mlflow.start_run(nested=True):

        # pre-processing steps
        standardizer = StandardScaler()

        # define model step
        skrf_regressor = RandomForestRegressor(
        bootstrap=True,
        criterion="friedman_mse",
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=746710809,
        )

        # create pipeline of preprocessing steps + model
        pipeline = Pipeline([
          ("standardizer", standardizer),
          ("regressor", skrf_regressor),
         ])

        pipelineModel = pipeline.fit(X_train, y_train)

        # Evaluate and log metrics
        predicted_probs = pipelineModel.predict(X_test)
        rmse = sklearn.metrics.mean_squared_error(y_test, predicted_probs, squared=False)

        mlflow.log_metric('rmse', rmse)

    return rmse

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the objective function
# MAGIC Here it takes the training function and passes the parameters to it (once formatted) and outputs the loss function so HyperOpt can use it. 

# COMMAND ----------

# Define hyperopt objective function
def train_with_hyperopt(params):

    # Some hyperparams take intger only
    min_samples_leaf = int(params['min_samples_leaf'])
    min_samples_split = int(params['min_samples_split'])
    max_features = params['max_features']
    # pass hyperparams into defined model
    rmse = train_model(max_features, min_samples_split, min_samples_leaf)

    # minimse rmse
    loss = rmse
    return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define the search space
# MAGIC Here the search space parameters can be defined in the following way:
# MAGIC
# MAGIC -  'uniform' - for real parameters with uniform weighting
# MAGIC -  'quniform' - for integer parameters with uniform weighting
# MAGIC -  'choice' - discrete parameters chosen from a given list
# MAGIC -  'lognormal' - real parameter weighted so the log of the return is normally distributed
# MAGIC This is the grid space that will be searched. 
# MAGIC

# COMMAND ----------

# Define search space
space = {
    'max_features': hp.uniform('max_features', 0, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, q=1),
    'min_samples_leaf': hp.quniform('min_samples_leaf', 2, 10, q=1),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running HyperOpt
# MAGIC The run ready to be exectued with the command below (fmin). This is where is searches for the best parameter that minimise the loss function. The objective function and grid space are passed as arguments. The 'max_evals' limits the number of iterations over the cluster. The output is a dictionary containing the best parameters from the run. 

# COMMAND ----------

# Run HyperOpt to find optimal hyperparam values
from hyperopt import SparkTrials

spark_trials = SparkTrials(parallelism=4)

with mlflow.start_run():
    best_params = fmin(
        fn=train_with_hyperopt,
        space=space,
        algo=tpe.suggest,
        max_evals=40,
        trials = spark_trials
    )

# COMMAND ----------

best_params
