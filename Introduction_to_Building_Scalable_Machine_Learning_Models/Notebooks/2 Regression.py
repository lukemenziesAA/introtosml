# Databricks notebook source
# MAGIC %md
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px; padding-left:150px">
# MAGIC   <img src="https://static1.squarespace.com/static/5bce4071ab1a620db382773e/t/5d266c78abb6d10001e4013e/1562799225083/appliedazuredatabricks3.png" alt="Databricks Learning" style="width: 600px; height: 163px">
# MAGIC </div>
# MAGIC
# MAGIC # *Introduction to Regression*. Presented by <a href="www.advancinganalytics.co.uk">Advancing Analytics</a>

# COMMAND ----------

# MAGIC %md
# MAGIC 💡 In this lesson you will grasp the following concepts:
# MAGIC * How to build a linear regression model
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading in the data
# MAGIC Below is the command for reading in a csv file into a Spark Dataframe that can be used passed to a model. The dataframe can be displayed using 'display(df)'. Or it can be converted to a Pandas Dataframe and displayed by typeing 'df' into a cell.

# COMMAND ----------

readPath = 'dbfs:/mnt/azureml/real_estate.csv'

df = (spark
      .read
      .option('header', True)
      .option("inferSchema",True)
      .format('csv')
      .load(readPath)
     )

label = "Y house price of unit area"

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import routines for building up the model

# COMMAND ----------

from pyspark.ml.feature import VectorIndexer, VectorAssembler
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

# MAGIC %md
# MAGIC ## Split the dataframe into a test and train dataset
# MAGIC The cell below shows the splitting of data into a training and test dataset. This is a vital step when building  a machine learninging model since a model needs to be tested on data it hasn't seen before. Typically the test dataset if smaller than the training dataset (20%-30%). Depending on the type of problem, will determine the way it is split. Below uses a random split where the data is shuffled around (removes ordering).

# COMMAND ----------

seed = 42
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=seed)

print("We have %d training examples and %d test examples." % (trainDF.count(), testDF.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the vector assembler and the model
# MAGIC Below shows the vector assember and the linear regression model being generated. The vector assembler creates a vectorized column that contains all the features. This needs to done to before passing the dataframe to the regression model. Otherwise and error will be raised. The LinearRegression model takes the featuresCol as an argument (the vectorized column containing all the features), as well as which column is the label column. This differs from the Scikit-learn library convention where the label data is passed in a separate argument. 

# COMMAND ----------

feats = df.drop(label).columns
vectorAssembler = VectorAssembler(inputCols=feats, outputCol='rawFeatures', handleInvalid='skip')

model = LinearRegression(featuresCol = 'rawFeatures', labelCol=label, maxIter=10)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Create a pipeline
# MAGIC The cell below is used to setup the pipeline for a machine learning model. A pipeline allows the user to orchastrate each step for a model including preparation, transformations and training. They are also a good tool to prevent data leakage that can happen in some transformation steps if not done correctly. 

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages([vectorAssembler, model])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Training and predicting
# MAGIC Below's command trains the model with the training dataset and uses the test set to generate a prediction, which can be compare with the test data's actual results since the model wouldn't have seen any of the test data in training. There's two ways of doing this and we will demonstrate both ways:

# COMMAND ----------

# MAGIC %md
# MAGIC ### (1) Train, save and predict
# MAGIC

# COMMAND ----------

pipeline_model = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **Saving Models**
# MAGIC
# MAGIC We can save our models to persistent storage (e.g. DBFS) in case our cluster goes down so we don't have to recompute our results.

# COMMAND ----------

modelPath = 'dbfs:/mnt/azureml/reg_model'

# COMMAND ----------

pipeline_model.write().overwrite().save(modelPath)

# COMMAND ----------

# MAGIC %md
# MAGIC **Loading models**
# MAGIC
# MAGIC When you load in models, you need to know the type of model you are loading back in (was it a linear regression or logistic regression model?).
# MAGIC For this reason, we recommend you always put your transformers/estimators into a Pipeline, so you can always load the generic PipelineModel back in.
# MAGIC

# COMMAND ----------

from pyspark.ml import PipelineModel

saved_pipeline_model = PipelineModel.load(modelPath)

# COMMAND ----------

# MAGIC %md
# MAGIC **Predict**
# MAGIC
# MAGIC Apply model to test set

# COMMAND ----------

pred_df = saved_pipeline_model.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### (2) Train and predict
# MAGIC

# COMMAND ----------

predictionDF = pipeline.fit(trainDF).transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring the model
# MAGIC The cell below shows how the use can score the model. A couple of metrics are shown below. More information on scoring metrics will be given in another part of the course. 

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluatorrmse = RegressionEvaluator().setMetricName("rmse").setPredictionCol("prediction").setLabelCol(label)
evaluatorR2 = RegressionEvaluator().setMetricName("r2").setPredictionCol("prediction").setLabelCol(label)

rmse = evaluatorrmse.evaluate(predictionDF)
r2 = evaluatorR2.evaluate(predictionDF)
print("Test RMSE = %f" % rmse)
print("Test R2 = %f" % r2)
