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
#Full_data['max_kills_by_team'] = Full_data.groupby('groupId').kills.transform('max')
Full_data['total_team_damage'] = Full_data.groupby('groupId').damageDealt.transform('sum')
Full_data['total_team_kills'] =  Full_data.groupby('groupId').kills.transform('sum')
#Full_data['total_team_items'] = Full_data.groupby('groupId').total_items_acquired.transform('sum')
Full_data['team_kill_points'] = Full_data.groupby('groupId').killPoints.transform('sum')
Full_data['team_kill_rank'] = Full_data.groupby('groupId').killPlace.transform('mean')
Full_data['team_normal_rank'] = Full_data.groupby('groupId').rankPoints.transform('mean')
#Full_data['totalDistance'] = Full_data['rideDistance'] + Full_data['walkDistance'] + Full_data['swimDistance']
Full_data = Full_data.drop(columns = ['Id','groupId','matchId', 'roadKills','numGroups'])
#Full_data = Full_data.drop(columns = ['Id','groupId','matchId', 'roadKills','numGroups','rideDistance','walkDistance','swimDistance'])
#########################################################################

sample_df = Full_data.sample(frac=0.01, replace=True).reset_index()
sample_df = sample_df.drop(columns = ['index'])
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
    feature.PCA(k=28, inputCol='zfeatures', outputCol='loadings')
]).fit(pj_sp_df)
###########################################################
result = pipeline_PCA.transform(pj_sp_df).select("loadings")
result.show(1)
####3######

principal_components = pipeline_PCA.stages[-1].pc.toArray()
pca_model = pipeline_PCA.stages[-1]
explainedVariance = pca_model.explainedVariance
pca_df = pipeline_PCA.transform(pj_sp_df).collect()
##
for i in range(0,1000):
    if explainedVariance[i] < 0.01:
        break
print ('best k = ', i-1)

sumEV = 0
for i in range(0,1000):
    sumEV += explainedVariance[i]
    if sumEV > 0.98:
        break
print ('best k = ', i-1)
##
plt.plot(np.cumsum(explainedVariance))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
############## Reduced Dimension #############################
df = pj_sp_df.drop('winPlacePerc')
sp_np = df.toPandas().values
pc_df = np.dot(sp_np,principal_components[:,:17])
##########################################################
##########################################################
#for i in range(np.size(principal_components,1)):
pc1_pd =pd.DataFrame({'Feature': inputcol, 'abs_loading': abs(principal_components[:,0])}).sort_values('abs_loading',ascending=False)
pc2_pd =pd.DataFrame({'Feature': inputcol, 'abs_loading': abs(principal_components[:,1])}).sort_values('abs_loading',ascending=False)


PCA_choosed = principal_components[:,:17]
#############
training_df, validation_df, testing_df = pj_sp_df.randomSplit([0.6, 0.3, 0.1], seed=0)

## Linear with Origin
# RMSE Function
rmse = fn.sqrt(fn.mean((fn.col('winPlacePerc')-fn.col('prediction'))**2)).alias("rmse")
def rmseLr(alpha,beta,maxiter = 100,fit_df = training_df, transform_df=validation_df):
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
      ]).fit(fit_df)

    coeff = pipe_Lr_og.stages[2].coefficients.toArray()
    coeffs_df = pd.DataFrame({'Features': inputcol, 'Coeffs': abs(coeff)})
    coeffs_df.sort_values('Coeffs', ascending=False)
    rmse3_df = pipe_Lr_og.transform(transform_df).select(rmse)
    return {'rmse':rmse3_df.show(),'pipe_model':pipe_Lr_og}

### Model 1 
rmseLr(0,0)['rmse']
### Model 2 
rmseLr(1,0.1)['rmse']
### Model 3 
rmseLr(0,0.1)['rmse']
### Model 4 
rmseLr(0.5,0.1)['rmse']

############### Model 1 is the best
### Model 1
rmseLr(0,0)['pipe_model'].transform(testing_df).select(rmse).show()
