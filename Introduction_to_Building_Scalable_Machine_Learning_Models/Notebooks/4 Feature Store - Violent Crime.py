# Databricks notebook source
# MAGIC %md
# MAGIC ### Feature Store Demo
# MAGIC
# MAGIC This demonstrates the setup of a feature store

# COMMAND ----------

# Import libraries
from databricks.feature_store import feature_table, FeatureStoreClient, FeatureLookup
from pyspark.sql.functions import col, when

# COMMAND ----------

# Read in data
data = (
    spark.read.format("parquet")
    .option("inferSchema", True)
    .option("header", True)
    .load("/mnt/training/crime-data-2016/Crime-Data-Boston-2016.parquet")
    .drop_duplicates(subset=["INCIDENT_NUMBER", "OCCURRED_ON_DATE"])
)

display(data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Feature 
# MAGIC Here we will create a feature which has a value of 1 one if a violent crime has been commited, and 0 if not. 

# COMMAND ----------

# Column list
violent_crimes = [
    "Simple Assault",
    "Indecent Assault",
    "Aggravated Assault",
    "Homicide",
    "HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE",
]
# List of all columns
cols = [
    "INCIDENT_NUMBER",
    "OCCURRED_ON_DATE",
    "REPORTING_AREA",
    "DISTRICT",
    "YEAR",
    "MONTH",
    "DAY_OF_WEEK",
    "HOUR",
    "UCR_PART",
    "LATITUDE",
    "LONGITUDE",
    "Violent_Crime",
]

# COMMAND ----------

# Here we will create a new column in the data, Violent_Crime containing a value of one if the OFFENSE_CODE_GROUP matches the list of violent crimes
df = data.withColumn(
    "Violent_Crime",
    when(col("OFFENSE_CODE_GROUP").isin(violent_crimes), 1).otherwise(0),
).select(cols)

# COMMAND ----------

# MAGIC %md
# MAGIC Next we will create a database for violent crime data

# COMMAND ----------

# MAGIC %sql DROP DATABASE violent_crime CASCADE;

# COMMAND ----------

# MAGIC %sql CREATE DATABASE IF NOT EXISTS violent_crime

# COMMAND ----------

# MAGIC %md
# MAGIC # Store the Feature
# MAGIC Now that the feature has been created we need to create a table, and store the feature in it!

# COMMAND ----------

# Create feature store client
fs = FeatureStoreClient()

# COMMAND ----------

# Create table
try:
    vc_feature_table = fs.create_table(
        name="violent_crime.features",
        primary_keys=["INCIDENT_NUMBER", "OCCURRED_ON_DATE"],
        schema=df.schema,
        description="features",
    )
except ValueError as e:
    # If the table already exists, do nothing
    pass

# COMMAND ----------

# Write the features DataFrame to the feature store table
fs.write_table(
    name="violent_crime.features",
    df=df,
    mode="overwrite",
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train a model using the new feature

# COMMAND ----------

from databricks.feature_store import feature_table, FeatureStoreClient, FeatureLookup
from pyspark.sql.functions import col, when

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import mlflow

# COMMAND ----------

# Read in the data to be used for training
data = (
    spark.read.format("parquet")
    .option("inferSchema", True)
    .option("header", True)
    .load("/mnt/training/crime-data-2016/Crime-Data-Boston-2016.parquet")
    .drop_duplicates(subset=["INCIDENT_NUMBER", "OCCURRED_ON_DATE"])
)

# COMMAND ----------

# Create a feature lookup
crime_feature_lookup = [
    FeatureLookup(
        table_name="violent_crime.features",
        feature_names=[
            "REPORTING_AREA",
            "DISTRICT",
            "YEAR",
            "MONTH",
            "DAY_OF_WEEK",
            "HOUR",
            "UCR_PART",
        ],
        lookup_key=["INCIDENT_NUMBER", "OCCURRED_ON_DATE"],
    ),
]

# COMMAND ----------

# Create a training set using Feature Store
train = fs.create_training_set(
    df.select(["INCIDENT_NUMBER", "OCCURRED_ON_DATE", "Violent_Crime"]),
    feature_lookups=crime_feature_lookup,
    label="Violent_Crime",
)
train_1 = train.load_df().toPandas()

# COMMAND ----------

# Create an xgboost model
model = XGBClassifier()

# Create the training set and labels
X = train_1.drop(["Violent_Crime"], axis=1)
y = train_1["Violent_Crime"]

# COMMAND ----------

# Create a column transformer to encode some of the variables
trans = ColumnTransformer(
    [
        (
            "cats",
            OneHotEncoder(),
            ["YEAR", "DAY_OF_WEEK", "UCR_PART", "DISTRICT", "REPORTING_AREA"],
        )
    ]
)
# Create a pipeline for the model
pipe = Pipeline([("pre", trans), ("model", model)])

# COMMAND ----------

# Train the model
pipe.fit(X, y)

# COMMAND ----------

# Score the model
pipe.score(X, y)

# COMMAND ----------

# Log the trained model with MLflow and package it with feature lookup information.
fs.log_model(
    model=pipe,
    artifact_path="dbfs:/databricks/mlflow-tracking/2632811917223553/28788e64a4084a6280cfa7e45affc6b9/artifacts/model",
    flavor=mlflow.sklearn,
    training_set=train,
    registered_model_name="violentcrime",
)
