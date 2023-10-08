# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Imbalanced Datasets*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/mnt/azureml/credit_card_churn.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import col

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

label = "Attrition_Flag"
seed = 42
drop_cols = [
    "CLIENTNUM",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

X = df.drop(*drop_cols + [label])
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=seed)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Undersample the training dataset
# MAGIC Here is an example of how to undersample the dataset. Pyspark has no explicit method for undersampling so it has to be done in the following way. There are two training sets, the normal training set and the undersampled training set. 

# COMMAND ----------

major_df = trainDF.filter(col(label) == "Existing Customer")
minor_df = trainDF.filter(col(label) == "Attrited Customer")
ratio = float(major_df.count() / minor_df.count())
sampled_majority_df = major_df.sample(False, 1 / ratio)
trainDF_u = sampled_majority_df.unionAll(minor_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.sql.types import StringType
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StringIndexer
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline, training and predicting

# COMMAND ----------

cat_data = [i.name for i in X.schema if isinstance(i.dataType, StringType)]
num_data = [i.name for i in X.schema if i.name not in cat_data]
cat_feats_num = [i + "_cat_num" for i in cat_data]
cat_feats = [i + "_cat" for i in cat_data]
vec_feats = num_data + cat_feats
stringIngexer = StringIndexer(
    inputCols=cat_data + [label], outputCols=cat_feats_num + [label + "_num"]
)
ohe = OneHotEncoder(inputCols=cat_feats_num, outputCols=cat_feats)
vectorAssembler = VectorAssembler(
    inputCols=vec_feats, outputCol="rawFeatures", handleInvalid="skip"
)

model = LogisticRegression(
    featuresCol="rawFeatures", labelCol=label + "_num", maxIter=10
)

# COMMAND ----------

pipeline = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC Here two predictions are made. One using the undersampled training set and one using the normal training set. 

# COMMAND ----------

predictionDF = pipeline.fit(trainDF).transform(testDF)
predictionDF_u = pipeline.fit(trainDF_u).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model in PySpark

# COMMAND ----------

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("fMeasureByLabel")
    .setMetricLabel(1)
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorPrecision = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("precisionByLabel")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorRecall = (
    MulticlassClassificationEvaluator()
    .setMetricLabel(1.0)
    .setMetricName("recallByLabel")
    .setPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorrocauc = (
    BinaryClassificationEvaluator()
    .setMetricName("areaUnderROC")
    .setRawPredictionCol("prediction")
    .setLabelCol(label + "_num")
)
evaluatorPRauc = (
    BinaryClassificationEvaluator()
    .setMetricName("areaUnderPR")
    .setRawPredictionCol("prediction")
    .setLabelCol(label + "_num")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Below shows a comparison between the two models. It shows that the recall in the undersampled training is significantly higher as well as a better ROC-AUC score. 

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF_u)
ac = evaluatorac.evaluate(predictionDF_u)
rocauc = evaluatorrocauc.evaluate(predictionDF_u)
PRauc = evaluatorPRauc.evaluate(predictionDF_u)
precision = evaluatorPrecision.evaluate(predictionDF_u)
recall = evaluatorRecall.evaluate(predictionDF_u)

print("Undersampled")
print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
rocauc = evaluatorrocauc.evaluate(predictionDF)
PRauc = evaluatorPRauc.evaluate(predictionDF)
precision = evaluatorPrecision.evaluate(predictionDF)
recall = evaluatorRecall.evaluate(predictionDF)

print("Test F1 = %f" % f1)
print("Test Accuracuy = %f" % ac)
print("Test Precision = %f" % precision)
print("Test Recall = %f" % recall)
print("Test ROC-AUC = %f" % rocauc)
print("Test PR-AUC = %f" % PRauc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Imbalanced-learn to undersample
# MAGIC The library Imbalanced-learn, is a useful library for undersampling and oversampling. Below shows how to use SMOTE (oversampling) to generate a model. Below used the SMOTENC which allows for categorical data to be oversampled. Regular SMOTE cannot do this. 

# COMMAND ----------

df_pandas = df.toPandas()
X = df_pandas.drop(drop_cols + [label], axis=1)
y = df_pandas[label]

# COMMAND ----------

import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting the dataset into a test and train set

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=seed
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Imbalanced-learn pipeline
# MAGIC Imbalanced-learn requires a pipeline that is separate to Scikit-learn's pipeline. Otherwise SMOTENC will not work. 

# COMMAND ----------

transformer = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data)], remainder="passthrough"
)
full = transformer.fit_transform(X_train).shape[1]
cats_ind = full - (X_train.shape[1] - len(cat_data))
cats_ind = [i for i in range(cats_ind)]
pipeline_o = Pipeline(
    [
        ("trans", transformer),
        ("oversample", SMOTENC(cats_ind, random_state=seed)),
        ("model", LogisticRegression(max_iter=10000)),
    ]
)
pipeline = Pipeline(
    [("trans", transformer), ("model", LogisticRegression(max_iter=1000))]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and predict 

# COMMAND ----------

pipeline_o.fit(X_train, y_train)
pipeline.fit(X_train, y_train)
prediction_o = pipeline_o.predict(X_test)
prediction = pipeline.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification reports for oversampled model
# MAGIC Below displays the scores from the oversampled model and regular model. Below shows an improved recall with the oversampled model vs the regular model. There is also a slight improvement with the f1-score. 

# COMMAND ----------

from sklearn.metrics import (
    classification_report,
    plot_precision_recall_curve,
    plot_roc_curve,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Here shows the classification report being used on the test set. Remember to pass the labels used for the 0 and 1 classes, so they display. It is also recommended to output the results as a dictionary using 'output_dict=True'. 

# COMMAND ----------

classes = y.unique()
results_o = classification_report(
    y_test, prediction_o, labels=y.unique(), output_dict=True
)
results = classification_report(y_test, prediction, labels=y.unique(), output_dict=True)

# COMMAND ----------

pd.DataFrame(results_o)

# COMMAND ----------

pd.DataFrame(results)
