# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Scoring Metrics*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

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
X = df.drop(*(drop_cols + [label]))
y = df.select(label)
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=seed)

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

from pyspark.sql.types import StringType

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

predictionDF = pipeline.fit(trainDF).transform(testDF)

# COMMAND ----------

display(predictionDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the different classifiers in PySpark
# MAGIC Below shows how to use various classifiers in PySpark. Some come under multiclass and some binary. Multiclass evaluators don't have to be multi class to use them (they can be binary). Though binary evaluators cannot be used with multiclass models. Make sure to use 'setMetricLabel' and to put it to the label of interest. This should be set to 1 for binary classification. 

# COMMAND ----------

from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator,
)

evaluatorf1 = (
    MulticlassClassificationEvaluator()
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
# MAGIC ## Using Scikit-learn models and their evaluation routines
# MAGIC Scikit-learn has a very good variety of models. With them comes a variety of scoring metrics and evaluation routines for classification

# COMMAND ----------

df_pandas = df.toPandas()
X = df_pandas.drop(drop_cols + [label], axis=1)
y = df_pandas[label]

# COMMAND ----------

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC Splitting the dataset into a test and train set

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=seed
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Scikit-learn pipeline
# MAGIC Similar to PySpark, Scikit-learn has the means of using pipelines to ochestrate the steps. Along with a pipeline, Scikit-learn can use something called a column transformer to generate transformations for numerical and categorical columns. Here 'OneHotEncoder' has been used on categorical data. 

# COMMAND ----------

transformer = ColumnTransformer(
    [("cat", OneHotEncoder(), cat_data)], remainder="passthrough"
)
pipeline = Pipeline(
    [("trans", transformer), ("model", LogisticRegression(max_iter=1000))]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and predict 

# COMMAND ----------

pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification reports and plotting metrics
# MAGIC Scikit-learn has a number of useful evaluation routines for classification. The first one is a classification report. This can be used to provide a number of key metrics. It also generates useful plots such as the precision recall graph and the ROC curve. These routines are imported below. 

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
results = classification_report(y_test, prediction, labels=y.unique(), output_dict=True)

# COMMAND ----------

pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC Here are the two plotting routine for plotting the precision recall curves and the ROC curves. 

# COMMAND ----------

plot_precision_recall_curve(pipeline, X_test, y_test, pos_label=classes[1])
plot_roc_curve(pipeline, X_test, y_test, pos_label=classes[1])

# COMMAND ----------

# MAGIC %md
# MAGIC # Regression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data

# COMMAND ----------

readPath = "dbfs:/mnt/azureml/real_estate.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

label = "Y house price of unit area"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Splitting the dataset into a test and train set

# COMMAND ----------

trainDF, testDF = df.randomSplit([0.7, 0.3], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the model and pipeline

# COMMAND ----------

feats = df.drop(label).columns
vectorAssembler = VectorAssembler(
    inputCols=feats, outputCol="rawFeatures", handleInvalid="skip"
)


model = LinearRegression(featuresCol="rawFeatures", labelCol=label, maxIter=10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and generating prediction

# COMMAND ----------

pipeline = Pipeline().setStages([vectorAssembler, model])

# COMMAND ----------

predictionDF = pipeline.fit(trainDF).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using the different regressor evaluators in PySpark
# MAGIC Below shows how to use various evaluators in PySpark. Although there isn't as many for regression, there are three main one. R2, rmse and mae. 

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluatorrmse = (
    RegressionEvaluator()
    .setMetricName("rmse")
    .setPredictionCol("prediction")
    .setLabelCol(label)
)
evaluatorR2 = (
    RegressionEvaluator()
    .setMetricName("r2")
    .setPredictionCol("prediction")
    .setLabelCol(label)
)
evaluatormae = (
    RegressionEvaluator()
    .setMetricName("mae")
    .setPredictionCol("prediction")
    .setLabelCol(label)
)
rmse = evaluatorrmse.evaluate(predictionDF)
r2 = evaluatorR2.evaluate(predictionDF)
mae = evaluatormae.evaluate(predictionDF)
print("Test RMSE = %f" % rmse)
print("Test R2 = %f" % r2)
print("Test MAE = %f" % mae)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using Scikit-learn regression metrics
# MAGIC Below shows Scikit-learns metrics that are available for regression. The routines aren't as extensive as they are for classification, but there are a few to choose from. Below has shown some of the main ones. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert to a Pandas Dataframe

# COMMAND ----------

df_pandas = df.toPandas()
X = df_pandas.drop([label], axis=1)
y = df_pandas[label]

# COMMAND ----------

from sklearn.linear_model import LinearRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create test and train split

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=seed
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and predict on test set

# COMMAND ----------

model = LinearRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Scikit-learn models have a scoring metric native to each model. This can be accessed using 'model.score'. This is shown below. For regression models it is the 'r2' score and for classification models it as the 'accuracy' score. This is shown below. 

# COMMAND ----------

model.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Here some of the scoring metrics libraries are imported and used on the test set. An additional one to the PySpark library is 'mean_absolute_percentage_error'

# COMMAND ----------

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score

# COMMAND ----------

mse = mean_squared_error(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)
mape = mean_absolute_percentage_error(y_test, prediction)
r2 = r2_score(y_test, prediction)
print("Test MSE = %f" % mse)
print("Test R2 = %f" % r2)
print("Test MAE = %f" % mae)
print("Test MAPE = %f" % mape)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross validation
# MAGIC So far testing models has come about from splitting the dataset into a training and test set. A more robust method for scoring models is to use cross validation. This is where it take a selected number of even folds of the dataset (below 5 is selected) and it test on the smallest chunk of each fold. E.g as CV of 5 would test on 20% of the data. It test on all of the folds and trains on the remainder for each iteration. It outputs however many unique scores there are for however many fold chosen. In this case it is 5. Also note that some Scikit-learn regression scoring metrics are presented as negative. This is because it aligns with the direction of other metrics. I.e negative metrics small in value (for a good model) unless the sign is changed to negative. Then they become large in value relative to the other values on the negative side of the axis.  

# COMMAND ----------

scores = cross_val_score(
    model, X, y, cv=5, scoring="neg_mean_absolute_percentage_error"
)

# COMMAND ----------

print(scores)
print("Average MAPE CV score = {:.2f}%".format(-scores.mean() * 100))
