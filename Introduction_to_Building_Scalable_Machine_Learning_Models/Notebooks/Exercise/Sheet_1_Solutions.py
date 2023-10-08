# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Question sheet notebook*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

# COMMAND ----------

target_column = 'isFraud'
random_seed = 42

# COMMAND ----------

df = spark.read.format('parquet').load('dbfs:/mnt/azureml/transactions.parquet')

# COMMAND ----------

df_p = df.toPandas()

# COMMAND ----------

df_p.to_parquet('/dbfs/mnt/azureml/transactions_p.parquet')

# COMMAND ----------

display(df)

# COMMAND ----------

X = df.drop(target_column).toPandas()
y = df.select(target_column).toPandas()

# COMMAND ----------

X.currentExpDate = pd.to_datetime(X.currentExpDate)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_seed, test_size=0.2, shuffle=True)

# COMMAND ----------

def dummy_function(X):
    return X


cat_cols = [
    "acqCountry",
    "merchantCountryCode",
    "merchantCategoryCode",
    "transactionType",
]
date_cols = [
    "currentExpDate",
    "accountOpenDate",
    "dateOfLastAddressChange",
    "transactionDateTime",
]

pass_cols = ["merchantName"]

numeric_cols = list(
    dict(
        list(
            filter(
                lambda v: v[1] == "float64" or v[1] == "int32",
                X.dtypes.astype(str).to_dict().items(),
            )
        )
    ).keys()
)

num_pipeline = Pipeline(
    [("imp", SimpleImputer(fill_value=0))]
)

trans = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), cat_cols),
        ("num", num_pipeline, numeric_cols),
    ],
    sparse_threshold=0,
)

# COMMAND ----------

pipeline = Pipeline([('trans', trans) , ('model', XGBClassifier(objective="binary:logistic", use_label_encoder=False))])

# COMMAND ----------

pipeline.fit(X_train, y_train)

# COMMAND ----------

from sklearn.metrics import f1_score

# COMMAND ----------

y_pred = pipeline.predict(X_test)
score = f1_score(y_test, y_pred)

# COMMAND ----------

print("The F1 score is: {:.3f}".format(score))

# COMMAND ----------

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

date_pipeline = Pipeline(
    [
        ("get_feats", FunctionTransformer(loop_time_features)),
        ("ohe", OneHotEncoder()),
    ]
)

trans = ColumnTransformer(
    [
        ("date", date_pipeline, date_cols),
        ("cat", OneHotEncoder(), cat_cols),
        ("num", num_pipeline, numeric_cols),
    ],
    sparse_threshold=0,
)

# COMMAND ----------

new_X = trans.fit_transform(X)

# COMMAND ----------

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

X_T = transformer.fit_transform(X, y)

# COMMAND ----------

X_train_T, X_test_T, y_train_T, y_test_T = train_test_split(X_T, y, random_state=random_seed, test_size=0.2, shuffle=True)

# COMMAND ----------

pipeline = Pipeline([('model', XGBClassifier(objective="binary:logistic", use_label_encoder=False))])

# COMMAND ----------

pipeline.fit(X_train_T, y_train_T)

# COMMAND ----------

y_pred = pipeline.predict(X_test_T)
score = f1_score(y_test_T, y_pred)

# COMMAND ----------

print("The F1 score is: {:.3f}".format(score))

# COMMAND ----------

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# COMMAND ----------

cat_feats = [new_cols.index(i) for i in new_cols if i in date_labels + cat_labels]

# COMMAND ----------

transformer = Pipeline(
    [
        (
            "over_sample",
            SMOTENC(
                categorical_features=cat_feats,
                sampling_strategy=0.1,
                #n_jobs=-1,
                random_state=random_seed,
            ),
        ),
        (
            "under_sample",
            RandomUnderSampler(
                sampling_strategy=0.5, random_state=random_seed
            ),
        ),
    ]
)

# COMMAND ----------

X_train_T_T = transformer.fit_transform(X_train_T)

# COMMAND ----------

model = XGBClassifier(objective="binary:logistic", use_label_encoder=False)

# COMMAND ----------

#model.fit(X_train_T_T, y_train_t)

# COMMAND ----------

#y_pred = model.predict(X_test_T)
#score = f1_score(y_test_T, y_pred)

# COMMAND ----------

#print("The F1 score is: {:.3f}".format(score))

# COMMAND ----------

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

import mlflow

# COMMAND ----------

try:
    experiment = mlflow.create_experiment(name='/Users/luke@advancinganalytics.co.uk/fraud_experiment')
    mlflow.set_experiment(experiment_id=experiment)
except:
    experiment = mlflow.get_experiment_by_name(name='/Users/luke@advancinganalytics.co.uk/fraud_experiment')
    mlflow.set_experiment(experiment_id=experiment.experiment_id)

# COMMAND ----------

# Define search space
grid_space = {
    'max_depth': hp.quniform('max_depth', 3, 20, q=1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'n_estimators': hp.choice('n_estimators', [100, 200, 500]),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    }

# COMMAND ----------

# define model + hyperparameters to optimise
def train_model(max_depth, n_estimators, learning_rate, reg_alpha, reg_lambda):
    with mlflow.start_run(nested=True):


        model = XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda
                )


        model.fit(X_train_T_T, y_train_T)
        y_pred = pipeline.predict(X_test_T)
        score = f1_score(y_test_T, y_pred)
    return score

# COMMAND ----------

# Define hyperopt objective function
def train_with_hyperopt(params):
    # Some hyperparams take intger only
    max_depth = int(params['max_depth'])
    n_estimators = int(params['n_estimators'])
    learning_rate = params['learning_rate']
    reg_alpha = params['reg_alpha']
    reg_lambda = params['reg_lambda']
    # pass hyperparams into defined model
    f1 = train_model(max_depth, n_estimators, learning_rate, reg_alpha, reg_lambda)
    # minimse rmse
    loss = -f1
    return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

#test={'max_depth': 15, 'n_estimators':500, 'learning_rate':0.1, 'reg_lambda':0.1, 'reg_alpha': 0.2}
#train_with_hyperopt(test)

# COMMAND ----------

# Run HyperOpt to find optimal hyperparam values
from hyperopt import SparkTrials

spark_trials = SparkTrials()

with mlflow.start_run():
    best_params = fmin(
        fn=train_with_hyperopt,
        space=grid_space,
        algo=tpe.suggest,
        max_evals=40,
        trials = spark_trials
    )

# COMMAND ----------

print(best_params)
