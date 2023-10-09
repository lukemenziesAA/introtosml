# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC 
# MAGIC # *MLFLow*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC 
# MAGIC This notebook is intended to demonstrate the capabilities of MLFLow for monitoring machine learning models. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load in libraries

# COMMAND ----------

import mlflow
import pickle
from mlflow.tracking import MlflowClient
from mlflow.sklearn import eval_and_log_metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in the data into a Pandas Dataframe

# COMMAND ----------

readPath = #Fill in here

# Read in data
data = (
    spark.read.format("csv")
    .option("inferSchema", True)
    .option("header", True)
    .load(readPath)
    .toPandas()
)

label = "Attrition_Flag"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create experiment
# MAGIC Here the experiment where each run is logged is created. This doesn't explicitly have to be created but it makes a centralised 

# COMMAND ----------

try:
    experiment_id = mlflow.create_experiment(
        #Fill in here
    )
except:
    print("experiment already exists")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting the experiment
# MAGIC Once the experiment has been created, it should be set. Below sets the experiment and obtains the experiment id which can be passed to an MLFlow run, to ensure logging takes place in the selected experiment. 

# COMMAND ----------

# Set experiment
e_set = mlflow.set_experiment(
    experiment_name=#Fill in here
)
experiment_id = e_set.experiment_id

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scikit-learn autolog
# MAGIC MLFlow support autolog for multiple libraries. Below shows mlflow being set to autolog Scikit-learn routines. This saves having to manually log sklearn functions and routines. 

# COMMAND ----------

# Auto log scikit-learn models
#Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initialise model, set data and column transformer

# COMMAND ----------

# Initialise model
model = SVC()

drop_cols = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

X = data.drop(drop_cols + [label], axis=1)
y = data[label]

# COMMAND ----------

cat_data = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
num_data = [
    "Customer_Age",
    "Dependent_count",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

transformer = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data), ("scale", StandardScaler(), num_data)]
)

# COMMAND ----------

seed = 42
# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed, shuffle=True
)
X_train_trans = transformer.fit_transform(X_train)
X_test_trans = transformer.transform(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Store scaling parameters for transformation
# MAGIC The scale parameters are stored in a json file. This is so MLFlow can log them as artifacts. 

# COMMAND ----------

# Store scale parameters
scale_params = {}
scale = transformer.named_transformers_["scale"]
scale_params["scale"] = list(scale.scale_)
scale_params["var"] = list(scale.var_)
scale_params["mean"] = list(scale.mean_)

with open("scaling_params.json", "w") as file:
    json.dump(scale_params, file)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and scoring the model

# COMMAND ----------

# Train model
#Fill in here
score = model.score(X_test_trans, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Logging with MLFlow
# MAGIC The cell below demonstrates logging the model training and scoring using MLFlow. The experiment id is manually set. MLFlow can run without these commands in Databricks but it is good practice to manually set things. There are a number of ways MLFlow can log:
# MAGIC - mlflow.log_metric - model metrics or values
# MAGIC - mlflow.log_artifact - logging objects/artifacts. Note they need to be saved to a path and not passed directly
# MAGIC - mlflow.log_param - log model parameters
# MAGIC - mlflow.log_text - log strings from run
# MAGIC 
# MAGIC There are other but the main methods are listed above. When the run is finished, logging can be closed with 'mlflow.end_run()'.

# COMMAND ----------

# Run MLFLow on test set
with mlflow.start_run(experiment_id=experiment_id) as run:
    model.fit(X_train_trans, y_train)
    score = #Fill in here
    #log score
    #Fill in here
    y_pred = model.predict(X_test_trans)
    metrics = eval_and_log_metrics(model, X_test_trans, y_test, prefix="val_")
    #log multiple metrics
    #Fill in here
    #log scaling_params artifact
    #Fill in here
    run_id = run.info.run_id
mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Downloading artifacts
# MAGIC The user may wish to download an artifact. This model can be loaded locally say, using 'mlflow.sklearn.load_model'. Other items can be load that are packaged in the artifact. This can be used to create things like APIs. 

# COMMAND ----------

# Download model from client
client = MlflowClient()
outpath = client.download_artifacts(#Fill in here, ".")
dbutils.fs.cp("file:" + outpath, #Fill in here, True)
