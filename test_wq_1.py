# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:43:57 2019

@author: Qi Wang
"""

from pyspark.sql import SparkSession
import numpy as np

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

import pyspark
from pyspark.ml import feature, regression, Pipeline
from pyspark.sql import functions as fn, Row
from pyspark import sql
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression

import matplotlib.pyplot as plt
import pandas as pd

#########################################################################
Full_data = pd.read_csv('train_V2.csv', sep=',')
Full_data.describe()
Full_data.info()

#########################################################################
# Full_data[Full_data['winPlacePerc'].isnull()].index
Full_data = Full_data.dropna()
#########################################################################

sample_df = Full_data.sample(frac=0.01, replace=True).reset_index()
sample_df = sample_df.drop(columns = ['index','Id','groupId','matchId'])
sample_df['matchType'].unique()

###########
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
sample_df['matchType'] = sample_df['matchType'].map(mapper)
sample_df['winPlacePerc'] = sample_df['winPlacePerc'].astype('float')

pj_sp_df = spark.createDataFrame(sample_df)
##############################################################################
pj_sp_df = pj_sp_df.\
                    withColumn('solo',(fn.col('matchType') == 'solo').cast('int')).\
                    withColumn('duo',(fn.col('matchType') == 'duo').cast('int')).\
                    withColumn('squad',(fn.col('matchType') == 'squad').cast('int'))
pj_sp_df = pj_sp_df.drop('matchType')
pj_sp_df.show(5)                    
############################################################################3
inputcol = pj_sp_df.columns
inputcol.remove('winPlacePerc')
#################### pipeline
## PCA
pipeline_PCA = Pipeline(stages=[
    feature.VectorAssembler(inputCols=inputcol,outputCol='features'),
    feature.StandardScaler(withMean=True,inputCol='features', outputCol='zfeatures'),
    feature.PCA(k=19, inputCol='zfeatures', outputCol='loadings')
]).fit(pj_sp_df)
###########################################################
principal_components = pipeline_PCA.stages[-1].pc.toArray()
pca_model = pipeline_PCA.stages[-1]
explainedVariance = pca_model.explainedVariance
##
for i in range(0,1000):
    if explainedVariance[i] < 0.01:
        break
print ('best k = ', i)
##########################################################
#for i in range(np.size(principal_components,1)):
pc1_pd =pd.DataFrame({'Feature': inputcol, 'abs_loading': abs(principal_components[:,0])}).sort_values('abs_loading',ascending=False)
pc2_pd =pd.DataFrame({'Feature': inputcol, 'abs_loading': abs(principal_components[:,1])}).sort_values('abs_loading',ascending=False)

#############
training_df, validation_df, testing_df = pj_sp_df.randomSplit([0.6, 0.3, 0.1], seed=0)

## Linear with Origin
# RMSE Function
def rmseLr(alpha,beta,maxiter = 100):
    lr =  LinearRegression().\
        setLabelCol('winPlacePerc').\
        setFeaturesCol('scaledFeatures').\
        setRegParam(beta).\
        setMaxIter(maxiter).\
        setElasticNetParam(alpha)

    pipe_Lr_og = Pipeline(stages = [
      feature.VectorAssembler(inputCols = inputcol ,outputCol = 'feature'),
      feature.StandardScaler(withMean=True,inputCol="feature", outputCol="scaledFeatures"),
      lr
      #LogisticRegression(featuresCol = "scaledFeatures", labelCol = 'winPlacePerc')
      ]).fit(training_df)

    coeff = pipe_Lr_og.stages[2].coefficients.toArray()
    coeffs_df = pd.DataFrame({'Features': inputcol, 'Coeffs': abs(coeff)})
    coeffs_df.sort_values('Coeffs', ascending=False)


    rmse = fn.sqrt(fn.mean((fn.col('winPlacePerc')-fn.col('prediction'))**2)).alias("rmse")
    rmse3_df = pipe_Lr_og.transform(validation_df).select(rmse)
    return(rmse3_df.show())

### Model 1 
rmseLr(0,0)
### Model 2 
rmseLr(1,0.1)
### Model 3 
rmseLr(0,0.1)
### Model 4 
rmseLr(0.5,0.1)
