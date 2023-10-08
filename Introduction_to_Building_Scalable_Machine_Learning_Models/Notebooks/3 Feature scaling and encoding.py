# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Feature Scaling and Encoding*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = "dbfs:/mnt/azureml/credit_score_data/train_clean.csv"

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .format("csv")
    .load(readPath)
)

label = "Credit_Score"

# COMMAND ----------

display(df)

# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

pandas_df.to_csv('/databricks/driver/train_clean.csv', index=False)

# COMMAND ----------

dbutils.fs.ls('file:/databricks/driver/')

# COMMAND ----------

dbutils.fs.cp('File:/databricks/driver/train_clean.csv', 'dbfs:/mnt/azureml/train_clean.csv')

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

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setting up the pipeline with an imputer
# MAGIC The cell below used the PySpark imputer to within a pipeline to handle missing data. 

# COMMAND ----------

from pyspark.sql.types import StringType, IntegerType, DoubleType

# COMMAND ----------

cat_feats = [
    "Month",
    "Occupation",
    "Credit_Mix",
    "Payment_of_Min_Amount",
    "Payment_Behaviour",
]
num_feats = list(
    map(
        lambda y: y.name,
        filter(
            lambda x: isinstance(x.dataType, IntegerType)
            or isinstance(x.dataType, DoubleType),
            df.schema,
        ),
    )
)

out_cats = [i + "_catv" for i in cat_feats]
f_cats = [i + "_cat" for i in cat_feats]
vec_feats = num_feats + f_cats

stringIndexer = StringIndexer(
    inputCols=cat_feats + [label], outputCols=out_cats + ["Credit_Score_num"]
)

vectorAssembler = VectorAssembler(
    inputCols=vec_feats, outputCol="rawFeatures", handleInvalid="skip"
)
ohe = OneHotEncoder(inputCols=out_cats, outputCols=f_cats)

standardScaler = StandardScaler(inputCol="rawFeatures", outputCol="scaledFeatures")

# COMMAND ----------

fitted_index = stringIndexer.fit(df)
dfc = (
    fitted_index
    .transform(df)
    .select(num_feats + out_cats + ["Credit_Score_num"])
)
dfc = ohe.fit(dfc).transform(dfc).select(vec_feats + ["Credit_Score_num"])

# COMMAND ----------

display(dfc)

# COMMAND ----------

dfc = vectorAssembler.transform(dfc)
dfc = standardScaler.fit(dfc).transform(dfc)

# COMMAND ----------

display(dfc)

# COMMAND ----------

seed = 42
trainDF, testDF = dfc.sample(fraction=0.1).randomSplit([0.7, 0.3], seed=seed)

print(
    "We have %d training examples and %d test examples."
    % (trainDF.count(), testDF.count())
)

# COMMAND ----------

model = RandomForestClassifier(
    featuresCol="scaledFeatures", labelCol="Credit_Score_num"
)

# COMMAND ----------

fitted_model = model.fit(trainDF)

# COMMAND ----------

predictionDF = fitted_model.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Converting Indexes Back to String
# MAGIC The cell below demonstrates how to convert the prediction indexer back into the original string. The results are then displayed below. 

# COMMAND ----------

from pyspark.ml.feature import IndexToString

indtostring = IndexToString(inputCol="prediction", outputCol="prediction_string")
indtostring.setLabels(fitted_index.labelsArray[-1])

# COMMAND ----------

display(
    indtostring.transform(
        predictionDF.select("prediction")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC As is shown below, there is slightly worse improvement with imputation using the PySpark library. This can be due to RandomForestClassifier having its own method for handling missing data. Sometimes it is better to try without it if the model can handle missing entries. Another option is to use mode sophisticated imputation methods such as k-Nearest Neighbour Imputers. This isn't avaiable currently in PySpark ml. For this, one would use another library such as Scikit-learn. 

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluatorf1 = (
    MulticlassClassificationEvaluator()
    .setMetricName("fMeasureByLabel")
    .setMetricLabel(1)
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
# MAGIC ## Convert the to Pandas Dataframes
# MAGIC Here we are converting to Pandas Dataframes and creating training and test sets. 

# COMMAND ----------

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# COMMAND ----------

scale = StandardScaler()
ohe = OneHotEncoder(sparse=False)

# COMMAND ----------

df_pandas = df.toPandas()
X = df_pandas.set_index(["ID", "Customer_ID"]).drop(label, axis=1).fillna(0)
y = df_pandas.set_index(["ID", "Customer_ID"])[label]
X = scale.fit_transform(
    np.concatenate((ohe.fit_transform(X[cat_feats]), X[num_feats]), axis=1)
)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# COMMAND ----------

model = RandomForestClassifier()
model.fit(X_train, y_train)
