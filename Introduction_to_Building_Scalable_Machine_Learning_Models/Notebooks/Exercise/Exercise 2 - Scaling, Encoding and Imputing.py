# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Scaling, Encoding and Imputing*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = #Fill in here

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

label = "Credit_Score"

# COMMAND ----------

#Display the results
# Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import (
    VectorIndexer,
    VectorAssembler,
    StringIndexer,
    OneHotEncoder,
    StandardScaler,
)
from pyspark.ml.classification import RandomForestClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit(#Fill in here
)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline with an Scaling and OneHotEncoder
# MAGIC The cell below used the string indexer, one hot encoder, vector assembler and standard scalar. These can then be used within a pipeline (along with model), to create a model. Please note that the standard scalar goes last in the chain. 

# COMMAND ----------

cat_feats = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]
num_feats = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
    "Credit_History_Age (Years)",
]

out_cats = [i + "_catv" for i in cat_feats]
f_cats = [i + "_cat" for i in cat_feats]
vec_feats = num_feats + f_cats

inputCols = #Fill in here. Use categorical feature list and label column
outputCols = #Fill in here. Use out_cats list and add update label column
stringIngexer = StringIndexer(
    inputCols=inputCols, outputCols=outputCols
)

inputCols = #Fill in here
outputCols = #Fill in here
ohe = OneHotEncoder(inputCols=inputCols, outputCols=outputCols)

inputCols = # Fill in here
outputCols = #Fill in here
vectorAssembler = VectorAssembler(
    inputCols=inputCols, outputCol=outputCols, handleInvalid="skip"
)

inputCols = #Fill in here
outputCols = #Fill in here
standardScaler = StandardScaler(inputCol=inputCols, outputCol=outputCols)

labelCol = #Fill in here
model = LogisticRegression(
    featuresCol=outputCols, labelCol=labelCol, maxIter=10
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [#Fill in here]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

predictionDF = # Fill in here

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("f1")
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("F1 = %f" % f1)
print("Accuracuy = %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputing
# MAGIC Next an imputer can be created to be added to the pipeline. The imputer normally goes first within the pipeline. The impute strategy should be set. Select the appropriate strategy out of mean, median, mode. 

# COMMAND ----------

from pyspark.ml.feature import Imputer

# COMMAND ----------

imp_cols = [
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Num_Credit_Inquiries",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
]

inputCols = #Fill in here
outputCols = #Fill in here
imputer = Imputer(inputCols=inputCols, outputCols=outputCols)
imputer.setStrategy(#Fill in here)

# COMMAND ----------

from pyspark.ml import Pipeline

stages = [#Fill in here]
pipeline = Pipeline().setStages(stages)

# COMMAND ----------

predictionDF = # Fill in here

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring with imputing
# MAGIC Test the scoring again to see if imputing has improved the score. 

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("f1")
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol("Credit_Score_num")
)

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("F1 with imputing= %f" % f1)
print("Accuracuy with imputing= %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert the to Pandas Dataframes
# MAGIC Here we are converting to Pandas Dataframes and creating training and test sets. 

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore KNN imputing and Simple imputing with Scikit-learn
# MAGIC Below is an example of two pipelines used to create two versions of imputing with Scikit-learn. One with imputing, one without imputing. Here you can see the difference between the two imputing methods. A column transformer is used within the pipelines to convert the Pandas dataframes to the transformed version of the data needed for the models. 

# COMMAND ----------

transformer_knn = ColumnTransformer(
    [
        ("categories", OneHotEncoder(), cat_feats),
        ("imputation", KNNImputer(), num_feats),
    ]
)
transformer_simple = ColumnTransformer(
    [
        ("categories", OneHotEncoder(), cat_feats),
        ("imputation", SimpleImputer(), num_feats),
    ]
)
pipeline_simp_imp = Pipeline(
    [("trans", transformer_simple), ("model", RandomForestClassifier())]
)
pipeline_knn_imp = Pipeline(
    [("trans", transformer_knn), ("model", RandomForestClassifier())]
)

# COMMAND ----------

X_train = trainDF.toPandas().set_index(["ID", "Customer_ID"]).drop(label, axis=1).fillna(0)
y_train = trainDF.toPandas().select(label)
X_test = testDF.toPandas().set_index(["ID", "Customer_ID"]).drop(label, axis=1).fillna(0)
y_test = testDF.toPandas().select(label)

# COMMAND ----------

# MAGIC %md
# MAGIC Both scores can be compared, one with simple imutation, one with kNN imputation. 

# COMMAND ----------

pipeline_simp_imp.fit(X_train, y_train)
score = pipeline_simp_imp.score(X_test, y_test)
print("Score with Simple imputation {:.2f}".format(score))

# COMMAND ----------

pipeline_knn_imp.fit(X_train, y_train)
score = pipeline_knn_imp.score(X_test, y_test)
print("Score with kNN imputation {:.2f}".format(score))
