# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *SHAP*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

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

label = "Attrition_Flag"

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit(#Fill in here)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import libraries

# COMMAND ----------

from pyspark.ml.feature import (
    Imputer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
    OneHotEncoder,
)
from pyspark.ml.classification import DecisionTreeClassifier

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline, training and predicting

# COMMAND ----------

cat_feats = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]
num_feats = [
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

out_cats = [i + "_catv" for i in cat_feats]
f_cats = [i + "_cat" for i in cat_feats]
vec_feats = num_feats + f_cats

inputCols = #Fill in here
outputCols = #Fill in here
stringIngexer = StringIndexer(
    inputCols=inputCols,
    outputCols=outputCols,
)

inputCols = #Fill in here
outputCols = #Fill in here
ohe = OneHotEncoder(inputCols=inputCols, outputCols=outputCols)

inputCols = #Fill in here
outputCols = #Fill in here
vectorAssembler = VectorAssembler(
    inputCols=inputCols, outputCol=outputCols, handleInvalid="skip"
)

model = DecisionTreeClassifier(
    featuresCol=outputCols, labelCol=label + "_num"
)

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([stringIngexer, ohe, vectorAssembler, model])
trans = Pipeline().setStages([stringIngexer, ohe, vectorAssembler])
input_df = trans.fit(trainDF).transform(testDF)

# COMMAND ----------

pmodel = pipeline.fit(trainDF)
predictionDF = pmodel.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("f1")
    .setPredictionCol("prediction")
    .setLabelCol("Attrition_Flag_num")
)
evaluatorac = (
    MulticlassClassificationEvaluator()
    .setMetricName("accuracy")
    .setPredictionCol("prediction")
    .setLabelCol("Attrition_Flag_num")
)

f1 = evaluatorf1.evaluate(predictionDF)
ac = evaluatorac.evaluate(predictionDF)
print("Test F1 = %f" % f1)
print("Test Accuracu = %f" % ac)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use SHAP
# MAGIC Below is where we explore using SHAP values to explain the model. It should be noted that the data passed to SHAP cannot be as a PySpark dataframe. It has to be as a Pandas dataframe. We therefore have to convert it to the appropriate format for SHAP. This is done below. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Libraries

# COMMAND ----------

import shap
from typing import Iterator
import pandas as pd
from pyspark.sql.types import StructType, StructField, FloatType
from pyspark.sql.functions import size
from pyspark.ml.functions import vector_to_array

# COMMAND ----------

# MAGIC %md
# MAGIC The cell below converts the dataframe into the appropriate format and then converts it into a Pandas dataframe. Because PySpark categorical data is in vectorised form, it needs each element of the vector taken out and put into a separate column. The vector_to_array command can be used to do this. 

# COMMAND ----------

df_pandas = (
    input_df.select(
        num_feats
        + [
            vector_to_array(j + "_cat")[i].alias("{}_{}".format(j, i))
            for j in cat_feats
            for i in range(
                input_df.select(size(vector_to_array(j + "_cat"))).collect()[0][0]
            )
        ]
    )
).toPandas()

# COMMAND ----------

display(df_pandas)

# COMMAND ----------

#Train the model
rfc = #Fill in here
explainer = shap.TreeExplainer(rfc)
shap_values = explainer.shap_values(#Fill in here, check_additivity=False)

# COMMAND ----------

p = shap.summary_plot(shap_values[0], df_pandas, show=False)
display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scikit-learn and SHAP
# MAGIC Below we shall use the Scikit-learn library to craete a model. We can then compare the SHAP values using the PySpark model above with the Scikit-learn model below. You will notice that they're very similar in output. 

# COMMAND ----------

# Train a model
#Fill in here. Hint, no need to use a pipeline because the data is already transformed for df_pandas

# COMMAND ----------

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(df_pandas, check_additivity=False)

# COMMAND ----------

p = shap.summary_plot(shap_values[0], df_pandas, show=False)
display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using a UDF to create SHAP values
# MAGIC Generating SHAP values can be very computationally intensive. You can ustilise Sparks capabilities to speed up the process of generating SHAP values. Below is the code used to do it. 

# COMMAND ----------

explainer = shap.TreeExplainer(model)
columns_for_shap_calculation = df_pandas.columns


def calculate_shap(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for X in iterator:
        yield pd.DataFrame(
            explainer.shap_values(np.array(X), check_additivity=False)[0],
            columns=columns_for_shap_calculation,
        )


return_schema = StructType()
for feature in columns_for_shap_calculation:
    return_schema = return_schema.add(StructField(feature, FloatType()))

shap_values = (
    spark.createDataFrame(df_pandas)
    .mapInPandas(calculate_shap, schema=return_schema)
    .toPandas()
    .values
)

# COMMAND ----------

# MAGIC %md
# MAGIC Here you can see the displayed result is the same as the previous results that didn't use the UDF to speed up the process. 

# COMMAND ----------

p = #Fill in here
display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Displaying individual shap entries
# MAGIC Below displays how to use SHAP on individual predicted entries. 

# COMMAND ----------

shap_display = shap.force_plot(
    explainer.expected_value[0],
    shap_values[0],
    feature_names=df_pandas.columns,
    matplotlib=True,
)
display(shap_display)
