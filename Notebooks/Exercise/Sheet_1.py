# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC 
# MAGIC # *Question sheet notebook*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


import mlflow

# COMMAND ----------

target_column = 'isFraud'
random_seed = 42

# COMMAND ----------

# Load in the data 
df = # Fill in

# COMMAND ----------

# Display data and use Data Profile
# Fill in 

# COMMAND ----------

# Create X & y and convert to Pandas Dataframe
# Fill in
X = 
y = 

# COMMAND ----------

# Convert currentExpData column to datetime
# Fill in

# COMMAND ----------

# Split into test set and training set
X_train, X_test, y_train, y_test = # Fill in

# COMMAND ----------

def dummy_function(X):
    return X


# Categorical columns
cat_cols = [
    # Fill in
]

# Date columns
date_cols = [
    # Fill in
]


# Numerical columns
numeric_cols = # Fill in


trans = ColumnTransformer(
    [
        # Fill in 
    ],
    sparse_threshold=0,
)

# COMMAND ----------

# import classifier of your choosing
# Fill in

# COMMAND ----------

model = # Fill in 

# COMMAND ----------

try:
    experiment = mlflow.create_experiment(name=# Fill in)
    mlflow.set_experiment(experiment_id=experiment)
except:
    experiment = mlflow.get_experiment_by_name(name=# Fill in)
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

# COMMAND ----------

# Train model
# Fill in

# COMMAND ----------

# import scoring metric
# Fill in 

# COMMAND ----------

# get score
y_pred = # Fill in
score = # Fill in 

# COMMAND ----------

print("The *chosen metric* score is: {:.3f}".format(score))

# COMMAND ----------

# Functions for extracting datetime features
from datetime import datetime
from calendar import mdays

def week_of_month(tgtdate):
    days_this_month = mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    return (tgtdate - startdate).days // 7 + 1

def get_time_features(X):
    col = X.name
    year = X.dt.year
    month = X.dt.month
    dow = X.dt.dayofweek
    feats = [month, year, dow]
    feats_label = ["Month", "Year", "DayofWeek"]
    wim = X.apply(week_of_month)
    if len(wim.unique()) > 2:
        feats += [wim]
        feats_label += ["WeekinMonth"]
    time = X.dt.time
    if len(time.unique()) > 1:
        hour = X.dt.hour
        feats += [hour]
        feats_label += ["Hour"]
    feats_label = [f"{col}_{i}" for i in feats_label]
    out = pd.concat(tuple(feats), axis=1)
    out.columns = feats_label
    return out


def loop_time_features(X):
    ret = pd.DataFrame()
    for col in X.columns:
        out = get_time_features(X[col])
        ret = pd.concat((ret, out), axis=1)
    return ret

# COMMAND ----------

# Get datetime features 
date_pipeline = Pipeline(
    [
        # Fill in 
    ]
)

trans = ColumnTransformer(
    [
        ("date", date_pipeline, date_cols),
        # Categorical columns # Fill in
        # Numerical columns # Fill in
    ],
    sparse_threshold=0,
)

# COMMAND ----------

# Fit transformer
# Fill in

# COMMAND ----------

# Get the new column labels
check = loop_time_features(X[date_cols])
date_labels = check.columns.to_list()
date_ohe = trans.transformers_[0][1][1]
date_labels = date_ohe.get_feature_names(date_labels).tolist()
ohe = trans.transformers_[1][1]
cat_labels = ohe.get_feature_names(cat_cols).tolist()

new_cols = date_labels + cat_labels + numeric_cols

# COMMAND ----------

def reduce_size(X, features, columns=None):
    if isinstance(X, pd.DataFrame):
        return X[features]
    else:
        return pd.DataFrame(X, columns=columns)[features]

#kw_args = {"features": important_features, "columns": new_cols}

transformer = Pipeline(
    [
        ("trans", trans),
        #("reduce", FunctionTransformer(reduce_size, kw_args=kw_args)),
    ]
)

# COMMAND ----------

# Transform data
X_T = transformer. # Fill in

# COMMAND ----------

# Split to test set and training set
X_train_T, X_test_T, y_train_T, y_test_T = # Fill in

# COMMAND ----------

# Define model
# Fill in

# COMMAND ----------

# Train model
# Fill in 

# COMMAND ----------

# get score
y_pred = # Fill in
score = # Fill in 

# COMMAND ----------

print("The *chosen metric* score is: {:.3f}".format(score))

# COMMAND ----------

from imblearn.under_sampling import # Fill in 
from imblearn.over_sampling import # Fill in
from imblearn.pipeline import # Fill in

# COMMAND ----------

# Need to use if useing SMOTENC
#cat_feats = [new_cols.index(i) for i in new_cols if i in date_labels + cat_labels]

# COMMAND ----------

transformer = Pipeline(
    [
        (
            "over_sample",
            # Fill in
            ),
        (
            "under_sample",
            # Fill in
            ),
    ]
)

# COMMAND ----------

# Resample the data
# Fill in

# COMMAND ----------

# Define model of choice
# Fill in

# COMMAND ----------

# Train model
# Fill in 

# COMMAND ----------

# get score
y_pred = # Fill in
score = # Fill in 

# COMMAND ----------

print("The *chosen metric* score is: {:.3f}".format(score))

# COMMAND ----------

# import hyperopt routines
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# COMMAND ----------

# Define search space
grid_space = {
    # Fill in
    }

# COMMAND ----------

# define model + hyperparameters to optimise
# Fill in

# COMMAND ----------

# Define hyperopt objective function
# Fill in 

# COMMAND ----------

# Run HyperOpt to find optimal hyperparam values
from hyperopt import SparkTrials

# Fill in

# COMMAND ----------

# Print best parameters
# Fill in
