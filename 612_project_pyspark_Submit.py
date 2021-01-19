# Databricks notebook source
# MAGIC %md 
# MAGIC # Part 1 - Generation of dataset

# COMMAND ----------

pip install nltk

# COMMAND ----------

pip install mlflow

# COMMAND ----------

# MAGIC %md #####Loading the data which is already classified

# COMMAND ----------

# importing the datasets with ID and the classification type

df_jackrabbit = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/jackrabbit_classification_vs_type__1_.csv', header = True, multiLine = True, escape = '"')
df_httpclient = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/httpclient_classification_vs_type__1_.csv', header = True, multiLine = True, escape = '"')
df_lucene = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/lucene_classification_vs_type.csv', header = True, multiLine = True, escape = '"')

# COMMAND ----------

# MAGIC %md #####Load csv into pyspark dataframe, which includes the fetched summary along with the issue id

# COMMAND ----------

# loading the datasets which have summaries along with the issue IDs
df_summary_lucene = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_summary_lucene.csv', header = True, multiLine = True, escape = '"')
df_summary_httpclient = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_summary_httpclient.csv', header = True, multiLine = True, escape = '"')
df_summary_jackrabbit = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_summary_jackrabbit.csv', header = True, multiLine = True, escape = '"')

# COMMAND ----------

# MAGIC %md #####Joining the dataframes such that final dataframes include issue ID, summary and classification type

# COMMAND ----------

df_summary_lucene = df_summary_lucene.drop("_c0")
df_summary_httpclient = df_summary_httpclient.drop("_c0")
df_summary_jackrabbit = df_summary_jackrabbit.drop("_c0")

df_lucene_final = df_lucene.join(df_summary_lucene, df_lucene.ID == df_summary_lucene.ID).drop(df_summary_lucene.ID)
df_httpclient_final = df_httpclient.join(df_summary_httpclient, df_httpclient.ID == df_summary_httpclient.ID).drop(df_summary_httpclient.ID)
df_jackrabbit_final = df_jackrabbit.join(df_summary_jackrabbit, df_jackrabbit.ID == df_summary_jackrabbit.ID).drop(df_summary_jackrabbit.ID)

# dropping column which includes 'Type'
df_lucene_final = df_lucene_final.drop('TYPE')
df_httpclient_final = df_httpclient_final.drop('TYPE')
df_jackrabbit_final = df_jackrabbit_final.drop('TYPE')

# COMMAND ----------

# function for replacing all the values with the NUG which are not BUG
from pyspark.sql.types import *

def replaceNug(x):
  if x == 'BUG':
    return 'BUG'
  else:
    return 'NUG'

udf_nug = udf(replaceNug, StringType())

# COMMAND ----------

# Already classified dataframe with 'ID', 'SUMMARY', and 'CLASSIFIED' columns 
df_lucene_final = df_lucene_final.withColumn("CLASSIFIED", udf_nug('CLASSIFIED'))
df_httpclient_final = df_httpclient_final.withColumn("CLASSIFIED", udf_nug('CLASSIFIED'))
df_jackrabbit_final = df_jackrabbit_final.withColumn("CLASSIFIED", udf_nug('CLASSIFIED'))

# COMMAND ----------

# MAGIC 
# MAGIC %md #####Loading manually classified  issue reports 

# COMMAND ----------

df_httpclient_new = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_new_httpclient_data.csv', header = True, multiLine = True, escape = '"')
df_jcr_new = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_new_jcr_data.csv', header = True, multiLine = True, escape = '"')
df_lucene_new = spark.read.csv('dbfs:/FileStore/shared_uploads/neha.singh1@ucalgary.ca/df_new_lucene_data.csv', header = True, multiLine = True, escape = '"')

# dropping redundant columns
df_lucene_new = df_lucene_new.drop("_c0")
df_httpclient_new = df_httpclient_new.drop("_c0")
df_jcr_new = df_jcr_new.drop("_c0")

#removing rows with null values
df_jcr_new = df_jcr_new.na.drop()
df_lucene_new = df_lucene_new.na.drop()
df_httpclient_new = df_httpclient_new.na.drop()

# COMMAND ----------

# MAGIC %md #####Combining all 6 dataframes into single dataframe called df_issue_report 

# COMMAND ----------

from functools import reduce

df_lucene_http = reduce(lambda x,y:x.union(y), [df_lucene_new,df_httpclient_new])
df_lucene_http_jcr = reduce(lambda x,y:x.union(y), [df_lucene_http,df_jcr_new])
df_lucene_jcr_final = reduce(lambda x,y:x.union(y), [df_lucene_final,df_jackrabbit_final])
df_lucene_jcr_http_final = reduce(lambda x,y:x.union(y), [df_lucene_jcr_final,df_httpclient_final])
df_lucene_http_jcr = df_lucene_http_jcr.select("ID","CLASSIFIED" ,"SUMMARY")
df_issue_report = reduce(lambda x,y:x.union(y), [df_lucene_jcr_http_final,df_lucene_http_jcr])

# COMMAND ----------

df_issue_report.groupBy('CLASSIFIED').count().show()

# COMMAND ----------

# Data frame with the old data set
df_lucene_http_old = reduce(lambda x,y:x.union(y), [df_lucene_final,df_httpclient_final])
df_old = reduce(lambda x,y:x.union(y), [df_lucene_http_old,df_jackrabbit_final])
df_old.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Part 2 - Pre-processing the data

# COMMAND ----------

# MAGIC %md #####Removing noise from data

# COMMAND ----------

from pyspark.sql.functions import regexp_replace,col, trim, upper, lower

# function for pre-processing the data
def preProcessData(df, colName):
  
  # removing new line character
  df_line = df.withColumn('SUMMARY_PROCESSED', regexp_replace(col(colName), "[\\r\\n]", " "))
  
  # removing white
  df_line = df_line.withColumn("SUMMARY_PROCESSED",regexp_replace("SUMMARY_PROCESSED", "\\[.*\\]", " "))

  # converting the data to lower case
  df_lower = df_line.withColumn("SUMMARY_PROCESSED", lower(col("SUMMARY_PROCESSED")))

  # removing double quotes
  df_quotes = df_lower.withColumn("SUMMARY_PROCESSED", regexp_replace("SUMMARY_PROCESSED", '"', ""))

  # removing punctuation marks
  df_punc = df_quotes.withColumn("SUMMARY_PROCESSED", regexp_replace("SUMMARY_PROCESSED", '[.,/#!$%^&*;:{}=_\'~()?|]', " "))

  # removing hyphens
  df_hyphen = df_punc.withColumn("SUMMARY_PROCESSED", regexp_replace(col("SUMMARY_PROCESSED"), "-", " "))

  # removing integral values
  df_int = df_hyphen.withColumn("SUMMARY_PROCESSED", regexp_replace(col("SUMMARY_PROCESSED"), "\\d+", " "))
  
  # removing ids
  df_id = df_int.withColumn("SUMMARY_PROCESSED", regexp_replace(col("SUMMARY_PROCESSED"), "\[[a-z A-Z]+\\s\]", ""))
  
  #removing asf jira
  df_asf = df_id.withColumn("SUMMARY_PROCESSED", regexp_replace(col("SUMMARY_PROCESSED"), "asf jira", ""))

  # removing white
  df_space = df_asf.withColumn("SUMMARY_PROCESSED", regexp_replace(col("SUMMARY_PROCESSED"), "\\s+", " "))
  
  return df_space

# COMMAND ----------

# MAGIC %md ##### Tokenization of the summary

# COMMAND ----------

# tokenize_stop function tokenizes the column

from pyspark.ml.feature import Tokenizer

def tokenize_stop(df, colName):
  df_tokenizer = Tokenizer(inputCol = colName, outputCol = 'WORDS')
  df_tokenized = df_tokenizer.transform(df)
  return df_tokenized

# COMMAND ----------

# MAGIC %md #####Removing stop words

# COMMAND ----------

# list of stop words to be used

# list_articles = ['a', 'an', 'the']
# list_pronouns = [i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those']
# list_prepositions = ['of', 'at', 'by', 'for', 'with', 'in', 'out', 'on', 'off' ]
# list_conjuctions = ['and', 'but', 'or', 'yet', 'so', 'although', 'though']

list_stop_words = ['a', 'an', 'the','i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'of', 'at', 'by', 'for', 'with', 'in', 'out', 'on', 'off', 'and', 'but', 'or', 'yet', 'so', 'although', 'though']

# Function to write the stop word

from pyspark.sql.types import ArrayType, FloatType, StringType

def remove_stops(row):
    meaningful_words = [w for w in row if not w in list_stop_words]
    return meaningful_words
  
udf_remove_stops = udf(remove_stops, ArrayType(StringType()))

# COMMAND ----------

# MAGIC %md #####Stemming

# COMMAND ----------

# stemming of words
from pyspark.sql.functions import col, size
from nltk.stem.porter import *
from pyspark.sql.types import *

# Create user defined function for stemming with return type Array<String>
stemmer_udf = udf(lambda x: stem(x), ArrayType(StringType()))
# Instantiate stemmer object
stemmer = PorterStemmer()

# Create stemmer python function
def stem(in_vec):
    out_vec = []
    for t in in_vec:
        t_stem = stemmer.stem(t)
        if len(t_stem) > 2:
            out_vec.append(t_stem)       
    return out_vec

# COMMAND ----------

df_issue_report_processed = preProcessData(df_issue_report, "SUMMARY")
df_issue_report_tokenized = tokenize_stop(df_issue_report_processed, "SUMMARY_PROCESSED")
df_issue_report_stop = df_issue_report_tokenized.withColumn("WORDS_STOP",udf_remove_stops('WORDS'))
df_issue_report_stem = (df_issue_report_stop.withColumn("WORDS_STEM", stemmer_udf("WORDS_STOP")))

# COMMAND ----------

# data frame with ID, SUMMARY, CLASSIFIED, and WORD_STEM columns
columns = ['ID', 'SUMMARY', 'WORDS_STEM', 'CLASSIFIED']
df_clean = df_issue_report_stem.select(columns)
df_clean.show(10)

# COMMAND ----------

df_old.groupBy('CLASSIFIED').count().show()

# COMMAND ----------

df_clean.groupBy('CLASSIFIED').count().show()

# COMMAND ----------

# Pre-processing the old data set
df_old_processed = preProcessData(df_old, "SUMMARY")
df_old_tokenized = tokenize_stop(df_old_processed, "SUMMARY_PROCESSED")
df_old_stop = df_old_tokenized.withColumn("WORDS_STOP",udf_remove_stops('WORDS'))
df_old_stem = (df_old_stop.withColumn("WORDS_STEM", stemmer_udf("WORDS_STOP")))

# COMMAND ----------

# data frame with ID, SUMMARY, CLASSIFIED, and WORD_STEM columns
df_old_clean = df_old_stem.select(columns)
df_old_clean.groupBy('CLASSIFIED').count().show()

# COMMAND ----------

# MAGIC %md ## Part - 3 Splitting the data into train and test sets

# COMMAND ----------

#splitting the data
from sklearn.model_selection import train_test_split

# create training and testing datasets
train_data, test_data = df_clean.randomSplit([0.8,0.2], seed = 1234)

# COMMAND ----------

# creating training and testing data set from old dataset
train_old_data, test_old_data = df_old_clean.randomSplit([0.8, 0.2], seed = 4567)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### DTM generation of the training dataset

# COMMAND ----------

# MAGIC %md
# MAGIC minDF = 2 indicates that remove terms that occur in less than 2 documents
# MAGIC Here maxDf has default value of 1.0 which indicates that the term is only ignored if it occurs in all the documents

# COMMAND ----------

# count vectorization
from pyspark.ml.feature import CountVectorizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizerModel

stop_word_dtm = []

# creating a DTM for test dataset with only the words present in DTM of training data
def remove_stops_dtm(row):
    meaningful_words_dtm = [w for w in row if not w in stop_word_dtm]
    return meaningful_words_dtm
  
udf_remove_stops_dtm = udf(remove_stops_dtm, ArrayType(StringType()))


# converting lables BUG/NUG to 1/0 for machine learning models

def classifyNug(x):
  if x == 'BUG':
    return 1
  else:
    return 0

udf_classifyNug = udf(classifyNug, IntegerType())

def get_dtm_stopwords(list_test, list_train):
  # getting list of stop words for test dtm
#   stop_word_dtm = []
  for i in list_test:
    if i not in list_train:
      stop_word_dtm.append(i)
      
  return stop_word_dtm


def vectorizing_data(df_train, df_test, minDF):
  stop_word_dtm = []
  # vectorization of text
  preProcStages = []
  countVectorizer = CountVectorizer(minDF = minDF, inputCol='WORDS_STEM', outputCol='WORDS_TF')
  preProcStages += [countVectorizer]


  pipeline = Pipeline(stages=preProcStages)

  model = pipeline.fit(df_train)
  df_train_dtm = model.transform(df_train)
  
  stages = model.stages 
  vectorizers = [s for s in stages if isinstance(s, CountVectorizerModel)]
  x = [v.vocabulary for v in vectorizers]
  list_train_words = x[0]
  
  
  test_model = pipeline.fit(df_test)
  df_test_whole_dtm = test_model.transform(df_test)

  stages_whole = test_model.stages 
  vectorizers_whole = [s for s in stages_whole if isinstance(s, CountVectorizerModel)]
  x_whole = [v.vocabulary for v in vectorizers_whole]
  list_test_whole_words = x_whole[0]
  
  stop_word_dtm = get_dtm_stopwords(list_test_whole_words, list_train_words)
  
  
  df_test_stop = df_test.withColumn("WORDS_STEM",udf_remove_stops_dtm('WORDS_STEM'))
  
  df_test_dtm = vectorizers[0].transform(df_test_stop)
  
  df_train_dtm = df_train_dtm.withColumn("CLASSIFIED_LABEL",udf_classifyNug('CLASSIFIED'))
  df_test_dtm = df_test_dtm.withColumn("CLASSIFIED_LABEL",udf_classifyNug('CLASSIFIED'))

  df_train_new = df_train_dtm.select(['CLASSIFIED_LABEL', 'WORDS_TF', 'SUMMARY'])
  df_test_new = df_test_dtm.select(['CLASSIFIED_LABEL', 'WORDS_TF','SUMMARY'])
  
  return df_train_new, df_test_new

# COMMAND ----------

# MAGIC %md ####Normalisation and Vectorization

# COMMAND ----------

from pyspark.ml.linalg import SparseVector
from pyspark.ml.feature import Normalizer

df_train, df_test = vectorizing_data(train_data, test_data, 2)
df_train = Normalizer(inputCol="WORDS_TF", outputCol="WORDS_TF_norm", p=1).transform(df_train)
df_test = Normalizer(inputCol="WORDS_TF", outputCol="WORDS_TF_norm", p=1).transform(df_test)


# COMMAND ----------

# normalization and vectorization of old data
df_old_train, df_old_test = vectorizing_data(train_old_data, test_old_data, 2)
df_old_train = Normalizer(inputCol="WORDS_TF", outputCol="WORDS_TF_norm", p=1).transform(df_old_train)
df_old_test = Normalizer(inputCol="WORDS_TF", outputCol="WORDS_TF_norm", p=1).transform(df_old_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Part 4 - Machine Learning models

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.classification import NaiveBayes, LinearSVC
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.clustering import LDA
import numpy as np
from sklearn.metrics import average_precision_score
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# COMMAND ----------

# function for training the models
def trainModelAccuracy(model, metric, df_train, df_test):
  model_trained = model.fit(df_train)
  model_predictions = model_trained.transform(df_test)
  evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="CLASSIFIED_LABEL", metricName=metric)
  accuracy = evaluator.evaluate(model_predictions)
  return model_predictions, accuracy

# COMMAND ----------

def trainBestModel(model, paramGrid, metric, numFolds, df_train, df_test):
  evaluator = BinaryClassificationEvaluator(metricName = metric, labelCol = "CLASSIFIED_LABEL",rawPredictionCol="rawPrediction")
  cv = CrossValidator(estimator=model, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds = numFolds)
  cvModel = cv.fit(df_train)
  predictions = cvModel.transform(df_test)
  accuracy = evaluator.evaluate(predictions)
  return cvModel,accuracy

# COMMAND ----------

def getValues(df):
  TP = df.where("CLASSIFIED_LABEL == prediction and CLASSIFIED_LABEL = 1").count()
  TN = df.where("CLASSIFIED_LABEL == prediction and CLASSIFIED_LABEL = 0").count()
  FP = df.where("CLASSIFIED_LABEL != prediction and CLASSIFIED_LABEL = 0").count()
  FN = df.where("CLASSIFIED_LABEL != prediction and CLASSIFIED_LABEL = 1").count()
  return TP, TN, FP, FN

# COMMAND ----------

def precision(tp, fp):
  denom = tp+ fp
  if denom != 0:
    return tp/denom
  else:
    return 0

def recall(tp, fn):
  denom = tp+fn
  if denom != 0:
    return tp/denom
  else:
    return 0
  
def f1_score(precision, recall):
  denom = precision + recall
  if denom == 0:
    return 0
  else:
    return (2*precision*recall)/denom

def accuracy(tp, fp, tn, fn):
  r = tp + tn
  wrong = fp+fn
  total  = r+ wrong
  if total != 0:
    return r/total
  else:
    return 0

# COMMAND ----------

def printValues(df, model_name, helper, key):
  TP, TN, FP, FN = getValues(df)
  p = precision(TP, FP)
  r = recall(TP, FN)
  print("Precision for ",model_name," ", helper, " normalization for ",key, " dataset is : {:.2f} ".format(p))
  print("Recall for ",model_name," ", helper, " normalization for ",key, " dataset is : {:.2f} ".format(r))
  print("F1 score for ",model_name," ", helper, " normalization for ",key, " dataset is : {:.2f} ".format(f1_score(p,r)))
  print("Accuracy for ",model_name," ", helper," normalization for ",key, " dataset is : {:.2f} ".format(accuracy(TP, FP, TN, FN)))

# COMMAND ----------

# MAGIC %md
# MAGIC Creating train and test data for performing grid search

# COMMAND ----------

# create training and testing datasets
df_train_grid, df_test_grid = df_train.randomSplit([0.8,0.2], seed = 4444)

# COMMAND ----------

# MAGIC %md
# MAGIC Naive Bayes Model

# COMMAND ----------

# Naive bayes with hyper parameter
# Create ParamGrid and Evaluator for Cross Validation
nb_grid = NaiveBayes(featuresCol= 'WORDS_TF', labelCol= 'CLASSIFIED_LABEL')
paramGrid_test = ParamGridBuilder().addGrid(nb_grid.smoothing, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).build()
naive_best_model_test, naive_best_accuracy_test = trainBestModel(nb_grid, paramGrid_test, "areaUnderROC", 3, df_train_grid, df_test_grid)

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for Naive Bayes after normalization
naive_new = naive_best_model_test.bestModel.transform(df_test)
printValues(naive_new, "naive bayes", "without", "complete")

# COMMAND ----------

# Getting best naive parameter
print("Best smoothing parameter for naive bayes is ", naive_best_model_test.bestModel.getSmoothing())

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluating the models with best hyper parameters

# COMMAND ----------

#Naive bayes 
# create the trainer and set its parameters
nb = NaiveBayes(featuresCol= 'WORDS_TF', labelCol= 'CLASSIFIED_LABEL', modelType = "multinomial", smoothing = 1.0)
nb_norm = NaiveBayes(featuresCol= 'WORDS_TF_norm', labelCol= 'CLASSIFIED_LABEL', modelType = "multinomial", smoothing = 1.0)

nb_train,nb_accuracy = trainModelAccuracy(nb, "areaUnderROC", df_train, df_test)
nb_norm_train, nb_norm_accuracy = trainModelAccuracy(nb_norm, "areaUnderROC", df_train, df_test)

print ("Naive Bayes Model Accuracy without normalization: {:.2f} ".format(nb_accuracy))
print ("Naive Bayes Model Accuracy with normalization: {:.2f} ".format(nb_norm_accuracy))

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for Naive Bayes after normalization
printValues(nb_norm_train, "nave bayes", "with", "complete")

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for Naive Bayes before normalization
printValues(nb_train, "nave bayes", "without", "complete")

# COMMAND ----------

# naive bayes with the old data
#Naive bayes without smoothning 
# create the trainer and set its parameters
nb_old =  NaiveBayes(featuresCol= 'WORDS_TF', labelCol= 'CLASSIFIED_LABEL', modelType = "multinomial", smoothing = 1.0)
nb_old_norm = NaiveBayes(featuresCol= 'WORDS_TF_norm', labelCol= 'CLASSIFIED_LABEL', modelType = "multinomial", smoothing = 1.0)

nb_old_train,nb_old_accuracy = trainModelAccuracy(nb_old, "areaUnderROC", df_old_train, df_old_test)
nb_old_norm_train, nb_old_norm_accuracy = trainModelAccuracy(nb_old_norm, "areaUnderROC", df_old_train, df_old_test)

print ("Naive Bayes Model Accuracy without normalization: {:.2f} ".format(nb_old_accuracy))
print ("Naive Bayes Model Accuracy with normalization: {:.2f} ".format(nb_old_norm_accuracy))

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for Naive Bayes with normalization but old dataset
printValues(nb_old_norm_train, "naive bayes", "with", "old")

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for Naive Bayes without normalization but old dataset
printValues(nb_old_train, "naive bayes", "without", "old")

# COMMAND ----------

# MAGIC %md
# MAGIC Random Forest Classifier

# COMMAND ----------

# Training rfc based with regular and normalized features 
rfc = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF')
rfc_norm = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm')

rfc_train,rfc_accuracy = trainModelAccuracy(rfc, "areaUnderROC", df_train, df_test)
rfc_norm_train, rfc_norm_accuracy = trainModelAccuracy(rfc_norm, "areaUnderROC", df_train, df_test)

print ("Random Forest Model Accuracy without normalization: {:.2f} ".format(rfc_accuracy))
print ("Random Forest Model Accuracy with normalization: {:.2f} ".format(rfc_norm_accuracy))

# COMMAND ----------

printValues(rfc_train, "random forest", "without", "complete")

# COMMAND ----------

printValues(rfc_norm_train, "random forest", "with", "complete")

# COMMAND ----------

# RFC with hyper parameter
# Create ParamGrid and Evaluator for Cross Validation
paramGrid = ParamGridBuilder().addGrid(rfc.numTrees, [30, 40, 50, 60]).addGrid(rfc.maxDepth, [20, 30]).addGrid(rfc.maxBins, [30, 35, 40]).build()
rfc_best_model, rfc_best_accuracy = trainBestModel(rfc_norm, paramGrid, "areaUnderROC", 3, df_train_grid, df_test_grid)
print ("RFC Model Accuracy with normalization: {:.2f} ".format(rfc_best_accuracy))

# COMMAND ----------

print("Num of trees for RFC best model are  {:.2f} ".format(rfc_best_model.bestModel.getNumTrees))
print("Max depth for RFC best model are  {:.2f} ".format(rfc_best_model.bestModel.getMaxDepth()))
print("Max bins for RFC best model are  {:.2f} ".format(rfc_best_model.bestModel.getMaxBins()))


# COMMAND ----------

# training RFC on best hyper parameters
rfc_best = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF', numTrees= 40, maxDepth=30, maxBins=32)
rfc_best_train, rfc_best_accuracy = trainModelAccuracy(rfc_best, "areaUnderROC", df_train, df_test)
printValues(rfc_best_train, "random forest", "without", "complete")

# COMMAND ----------

# training RFC on best hyper parameters
rfc_best_test = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm', numTrees= 60, maxDepth=30, maxBins=40)
rfc_best_train_test, rfc_best_accuracy = trainModelAccuracy(rfc_best_test, "areaUnderROC", df_train, df_test)
printValues(rfc_best_train_test, "random forest", "with", "complete")

# COMMAND ----------

# training RFC on best hyper parameters without normalization
rfc_best_withoutNorm = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF', numTrees= 40, maxDepth=30, maxBins=32)
rfc_best_train_withoutNorm, rfc_best_accuracy = trainModelAccuracy(rfc_best_withoutNorm, "areaUnderROC", df_train, df_test)
printValues(rfc_best_train_withoutNorm, "random forest", "without", "complete")

# COMMAND ----------

# traininf RFC on old data set with bestparameters found
# training RFC on best hyper parameters without normalization with old dataset
rfc_best_old_withoutNorm = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF', numTrees= 40, maxDepth=30, maxBins=32)
rfc_best_old_train_withoutNorm, rfc_best_accuracy = trainModelAccuracy(rfc_best_old_withoutNorm, "areaUnderROC", df_old_train, df_old_test)
printValues(rfc_best_old_train_withoutNorm, "random forest", "without", "old")

# COMMAND ----------

# traininf RFC on old data set with bestparameters found
# training RFC on best hyper parameters with normalization with old dataset
rfc_best_old_withoutNorm = RandomForestClassifier(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm', numTrees= 40, maxDepth=30, maxBins=32)
rfc_best_old_train_withoutNorm, rfc_best_accuracy = trainModelAccuracy(rfc_best_old_withoutNorm, "areaUnderROC", df_old_train, df_old_test)
printValues(rfc_best_old_train_withoutNorm, "random forest", "with", "old")

# COMMAND ----------

# MAGIC %md ####Support Vector Machine

# COMMAND ----------

  
#lsvc = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF', maxIter=10)
#lsvc_norm = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm', maxIter=10)
lsvc = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF')
lsvc_norm = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm')

lsvc_train,lsvc_accuracy = trainModelAccuracy(lsvc, "areaUnderROC", df_train, df_test)
lsvc_norm_train, lsvc_norm_accuracy = trainModelAccuracy(lsvc_norm, "areaUnderROC", df_train, df_test)

print ("Support Vector Machine Model Accuracy without normalization: {:.2f} ".format(lsvc_accuracy))
print ("Support Vector Machine Accuracy with normalization: {:.2f} ".format(lsvc_norm_accuracy))

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for LSVC after normalization
printValues(lsvc_norm_train,"SVM", "with", "complete")


# COMMAND ----------

# precision, recall, f1 measure, and accuracy for LSVC before normalization
printValues(lsvc_train,"SVM", "without", "complete")

# COMMAND ----------

lsvc_old = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF')
lsvc_old_norm = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm')

lsvc_old_train,lsvc_old_accuracy = trainModelAccuracy(lsvc_old, "areaUnderROC", df_old_train, df_old_test)
lsvc_old_norm_train, lsvc_old_norm_accuracy = trainModelAccuracy(lsvc_old_norm, "areaUnderROC", df_old_train, df_old_test)

print ("Support Vector Machine Model Accuracy without normalization with old data: {:.2f} ".format(lsvc_old_accuracy))
print ("Support Vector Machine Accuracy with normalization with old data: {:.2f} ".format(lsvc_old_norm_accuracy))

# COMMAND ----------

#lsvc_old_norm_train.select('CLASSIFIED_LABEL','prediction','SUMMARY').show(100,truncate=False)

# COMMAND ----------

# precision, recall, f1 measure, and accuracy for LSVC before normalization
printValues(lsvc_old_train,"SVM", "without", "old")

# COMMAND ----------

printValues(lsvc_old_norm_train,"SVM", "with", "old")

# COMMAND ----------

# LSVC with hyper parameter
# Create ParamGrid and Evaluator for Cross Validation
paramGrid_lsvc = ParamGridBuilder().addGrid(LinearSVC.regParam, [10, 1, 0, 0.1, 0.01]).addGrid(LinearSVC.maxIter, [10, 20, 30]).build()
lsvc_best_model, lsvc_best_accuracy = trainBestModel(lsvc_norm, paramGrid_lsvc, "areaUnderROC", 3, df_train, df_test)
print ("Linear SVM Model Accuracy with normalization: {:.2f} ".format(lsvc_best_accuracy))

# COMMAND ----------

train_score = lsvc_best_model.transform(df_test)

# COMMAND ----------

printValues(train_score, "SVM", "with", "complete")

# COMMAND ----------

print ("LSVC Model max iterations are: {:.2f} ".format(lsvc_best_model.bestModel.getMaxIter()))
print ("LSVC Model regParam is: {:.2f} ".format(lsvc_best_model.bestModel.getRegParam()))

# COMMAND ----------

# MAGIC %md ###### LSVC with best hyper parameter for old dataset with normalisation

# COMMAND ----------


# training LSVC on best hyper parameters
lsvc_best_old_norm = LinearSVC(labelCol='CLASSIFIED_LABEL',featuresCol='WORDS_TF_norm', regParam= 0, maxIter=100)
lsvc_best_train_old, lsvc_best_accuracy_old = trainModelAccuracy(lsvc_best_old_norm, "areaUnderROC", df_old_train, df_old_test)
printValues(lsvc_best_train_old, "SVM", "with", "old")

# COMMAND ----------

# MAGIC %md ###### LSVC with best hyper parameter for old dataset without normalisation

# COMMAND ----------

# MAGIC %md ###### LSVC with best hyper parameter for old+new dataset with normalisation
