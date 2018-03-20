#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:15:33 2018

@author: jimgrund
"""

import os
### Provide the path here
os.chdir('/Users/jimgrund/Documents/GWU/machine learning/midterm_project/') 



from matplotlib import pyplot as plt
from sklearn import cross_validation
import scipy.stats as stats
import matplotlib.dates as mdates
import seaborn as sns
import gc
import pandas as pd
import numpy as np




### Load the SSA data into a pandas dataframe from CSV
# convert date columns to actual date type
# drop all columns we don't care about for this specific analysis
def initialize_data():
    ssa_df  = pd.read_csv('SSA-SA-MOWL.csv',header=None, index_col=False, names=['FileName','z', 'FileDate', 'RegionCode', 'StateCode', 'y', 'FullDate', 'FormattedDate',
                                                                'AllInitialReceipts', 'AllInitialPendingClosure', 'AllInitialDeterminations', 'AllInitialAllowances', 'AllInitialAllowanceRate',
                                                                'SSDIInitialReceipts', 'SSDIInitialPendingClosure', 'SSDIInitialDeterminations', 'SSDIInitialAllowances', 'SSDIInitialAllowanceRate',
                                                                'SSIInitialReceipts', 'SSIInitialPendingClosure', 'SSIInitialDeterminations', 'SSIInitialAllowances', 'SSIInitialAllowanceRate',
                                                                'ConcInitialReceipts', 'ConcInitialPendingClosure', 'ConcInitialDeterminations', 'ConcInitialAllowances', 'ConcInitialAllowanceRate',
                                                                'SSIDisabledChildInitialReceipts', 'SSIDisabledChildInitialPendingClosure', 'SSIDisabledChildInitialDeterminations', 'SSIDisabledChildInitialAllowances', 'SSIDisabledChildInitialAllowanceRate',
                                                                'PrototypeState',
                                                                'AllReconsReceipts', 'AllReconsPendingClosure', 'AllReconsDeterminations', 'AllReconsAllowances', 'AllReconsAllowanceRate',
                                                                'SSDIReconsReceipts', 'SSDIReconsPendingClosure', 'SSDIReconsDeterminations', 'SSDIReconsAllowances', 'SSDIReconsAllowanceRate',
                                                                'SSIReconsReceipts', 'SSIReconsPendingClosure', 'SSIReconsDeterminations', 'SSIReconsAllowances', 'SSIReconsAllowanceRate',
                                                                'ConcReconsReceipts', 'ConcReconsPendingClosure', 'ConcReconsDeterminations', 'ConcReconsAllowances', 'ConcReconsAllowanceRate',
                                                                'SSIDisabledChildReceipts', 'SSIDisabledChildPendingClosure', 'SSIDisabledChildDeterminations', 'SSIDisabledChildAllowances', 'SSIDisabledChildAllowanceRate',
                                                                'AllCDRReceipts', 'AllCDRPendingClosure', 'AllCDRDeterminations', 'AllCDRAllowances', 'AllCDRAllowanceRate',
                                                                'SSDICDRReceipts', 'SSDICDRPendingClosure', 'SSDICDRDeterminations', 'SSDICDRAllowances', 'SSDICDRAllowanceRate',
                                                                'SSICDRReceipts', 'SSICDRPendingClosure', 'SSICDRDeterminations', 'SSICDRAllowances', 'SSICDRAllowanceRate',
                                                                'ConcCDRReceipts', 'ConcCDRPendingClosure', 'ConcCDRDeterminations', 'ConcCDRAllowances', 'ConcCDRAllowanceRate',
                                                                'a','b','c','d','e','f','g','h','i','j','k','l','m','n'
                                                               ]
    )
    
    # add a year-only column to the dataframe
    ssa_df['Year'] = ssa_df['FormattedDate'].str.extract(r'([^\-]+)\-[^\-]+$', expand=True)
    # add a month-only column to the dataframe
    ssa_df['Month'] = ssa_df['FormattedDate'].str.extract(r'[^\-]+\-([^\-]+)$', expand=True)
    
    # convert date to date format
    ssa_df['FullDate'] = pd.to_datetime(ssa_df['FullDate'])
    
    # remove all columns not relevant to analyzing number of inbound applications and successful applications
    # by region, state, and date
    ssa_df.drop(['FileName','FileDate','z','y','a','b','c','d','e','f','g','h','i','j','k','l','m','n','PrototypeState',
                 'AllInitialPendingClosure','SSDIInitialPendingClosure','SSIInitialPendingClosure','ConcInitialPendingClosure','SSIDisabledChildInitialPendingClosure',
                 'AllInitialDeterminations','SSDIInitialDeterminations','SSIInitialDeterminations','ConcInitialDeterminations','SSIDisabledChildInitialDeterminations',
                 'AllReconsReceipts', 'AllReconsPendingClosure', 'AllReconsDeterminations', 'AllReconsAllowances', 'AllReconsAllowanceRate',
                 'SSDIReconsReceipts', 'SSDIReconsPendingClosure', 'SSDIReconsDeterminations', 'SSDIReconsAllowances', 'SSDIReconsAllowanceRate',
                 'SSIReconsReceipts', 'SSIReconsPendingClosure', 'SSIReconsDeterminations', 'SSIReconsAllowances', 'SSIReconsAllowanceRate',
                 'ConcReconsReceipts', 'ConcReconsPendingClosure', 'ConcReconsDeterminations', 'ConcReconsAllowances', 'ConcReconsAllowanceRate',
                 'SSIDisabledChildReceipts', 'SSIDisabledChildPendingClosure', 'SSIDisabledChildDeterminations', 'SSIDisabledChildAllowances', 'SSIDisabledChildAllowanceRate',
                 'AllCDRReceipts', 'AllCDRPendingClosure', 'AllCDRDeterminations', 'AllCDRAllowances', 'AllCDRAllowanceRate',
                 'SSDICDRReceipts', 'SSDICDRPendingClosure', 'SSDICDRDeterminations', 'SSDICDRAllowances', 'SSDICDRAllowanceRate',
                 'SSICDRReceipts', 'SSICDRPendingClosure', 'SSICDRDeterminations', 'SSICDRAllowances', 'SSICDRAllowanceRate',
                 'ConcCDRReceipts', 'ConcCDRPendingClosure', 'ConcCDRDeterminations', 'ConcCDRAllowances', 'ConcCDRAllowanceRate',
                 'FormattedDate'
                 ],axis=1,inplace=True)

    return(ssa_df)



# construct a new dataframe where the attributes we care about are now feature columns
# rather than every benefit type having it's own column, we will have a benefit type column 
def organize_data(dataframe):
    # construct a dataframe of SSI application types
    ssi_ssa_df = pd.DataFrame(dataframe[['RegionCode','StateCode','Year','Month','SSIInitialReceipts','SSIInitialAllowanceRate']])
    ssi_ssa_df['Type'] = 'SSI'
    ssi_ssa_df = ssi_ssa_df.rename(index=str, columns={"SSIInitialReceipts": "InitialReceipts", "SSIInitialAllowanceRate": "InitialAllowanceRate"})
    # construct a dataframe of SSDI application types
    ssdi_ssa_df = pd.DataFrame(dataframe[['RegionCode','StateCode','Year','Month','SSDIInitialReceipts','SSDIInitialAllowanceRate']])
    ssdi_ssa_df['Type'] = 'SSDI'
    ssdi_ssa_df = ssdi_ssa_df.rename(index=str, columns={"SSDIInitialReceipts": "InitialReceipts", "SSDIInitialAllowanceRate": "InitialAllowanceRate"})

    # construct a dataframe of SSIDC application types
    ssidc_ssa_df = pd.DataFrame(dataframe[['RegionCode','StateCode','Year','Month','SSIDisabledChildInitialReceipts','SSIDisabledChildInitialAllowanceRate']])
    ssidc_ssa_df['Type'] = 'SSIDC'
    ssidc_ssa_df = ssidc_ssa_df.rename(index=str, columns={"SSIDisabledChildInitialReceipts": "InitialReceipts", "SSIDisabledChildInitialAllowanceRate": "InitialAllowanceRate"})

    # construct a dataframe of CONC application types
    conc_ssa_df = pd.DataFrame(dataframe[['RegionCode','StateCode','Year','Month','ConcInitialReceipts','ConcInitialAllowanceRate']])
    conc_ssa_df['Type'] = 'Conc'
    conc_ssa_df = conc_ssa_df.rename(index=str, columns={"ConcInitialReceipts": "InitialReceipts", "ConcInitialAllowanceRate": "InitialAllowanceRate"})

    
    # append these dataframes together into one
    ssi_ssa_df = ssi_ssa_df.append(ssdi_ssa_df)
    ssi_ssa_df = ssi_ssa_df.append(ssidc_ssa_df)
    ssi_ssa_df = ssi_ssa_df.append(conc_ssa_df)
    
    # clean up 
    gc.collect()
    
    return(ssi_ssa_df)




# plot a 3x3 grid of AllowanceRate data for all the regions
def plot_regions(dataframe, regions):
    
    ####################################################
    # plot allowance rate graphs for regions
    
    #regions = ['ATL', 'BOS', 'CHI', 'DAL', 'KCM', 'NYC', 'PHL', 'SEA', 'SFO']
    ssa_df = dataframe
    
    # begin the plot
    fig = plt.figure(figsize=(15,15))
    xrow = 1
    yrow = 1
    plotnum = 0
    
    for region in regions:
        #print(region + ' Allowance Rate')
        
        dfquery = "RegionCode == '" +region + "'"
        #print(dfquery)
        df = ssa_df.query(dfquery).groupby(['FullDate'], as_index=False)['AllInitialAllowanceRate'].mean()
        x1 = df['FullDate']
        y1 = df['AllInitialAllowanceRate']
    
        if ( xrow >= 3 ):
            xrow = 1
            yrow +=1
        plotnum +=1
        
        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')

        x2 = mdates.date2num(x1)
    
        z=np.polyfit(x2,y1,1)
        p=np.poly1d(z)
    
        plt.plot(x1,p(x2),'r--') #add trendline to plot
    
        axes = plt.gca()
    
        axes.set_ylim([0,100])
    
        plt.title(region + ' Allowance Rate')
        plt.ylabel('% of apps approved')
        plt.xlabel('Date')
        
        xrow +=1
    
    
    fig = plt.gcf()
    plt.show()



# plot a 3x3 grid of AllowanceRate data for all the States within a region
def plot_region(dataframe, region, states):
        
    ####################################################
    # drill in and plot allowance graphs for PHL
    
    #states = ['DC','DE','VA','PA','EV','MD','WV']
    #region = 'PHL'
    plotnum = 1
    ssa_df = dataframe
    
    fig = plt.figure(figsize=(15,15))
    
    for state in states:
        #print(state + ' Allowance Rate')
        
        dfquery = "RegionCode == '" +region+ "' and StateCode == '" +state+ "'"
        #print(dfquery)
        df = ssa_df.query(dfquery).groupby(['FullDate'], as_index=False)['AllInitialAllowanceRate'].mean()
        x1 = df['FullDate']
        y1 = df['AllInitialAllowanceRate']
        
        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')
    
        x2 = mdates.date2num(x1)
    
        z=np.polyfit(x2,y1,1)
        p=np.poly1d(z)
    
        plt.plot(x1,p(x2),'r--') #add trendline to plot
    
        axes = plt.gca()
    
        axes.set_ylim([0,100])
    
        plt.title(state + ' Allowance Rate')
        plt.ylabel('% of apps approved')
        plt.xlabel('Date')
        
        plotnum +=1
    
  
    fig = plt.gcf()
    plt.show()





# plot a grid of the three different benefit types for a specific state
def plot_types(dataframe, region, state):

    ####################################################
    # drill in on types of assistance for VA
    # Social Security Disability Insurance benefits (SSDI), and Supplemental Security Income (SSI)
    types = ['SSI','SSDI','SSIDisabledChild']
    #region = 'PHL'
    #state  = 'VA'
    #SSDIInitialAllowanceRate
    plotnum = 1
    ssa_df = dataframe
    
    fig = plt.figure(figsize=(15,15))
    
    for type in types:
        #print(state + ': ' + type + ' Allowance Rate')
        
        dfquery = "RegionCode == '" +region+ "' and StateCode == '" +state+ "'"
        #print(dfquery)
        parameter = type+"InitialAllowanceRate"
        df = ssa_df.query(dfquery).groupby(['FullDate'], as_index=False)[parameter].mean()
        x1 = df['FullDate']
        y1 = df[parameter]

        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')
    
        x2 = mdates.date2num(x1)
    
        z=np.polyfit(x2,y1,1)
        p=np.poly1d(z)
    
        plt.plot(x1,p(x2),'r--') #add trendline to plot
    
        axes = plt.gca()
    
        axes.set_ylim([0,100])
    
        plt.title(state +": "+ type + ' Allowance Rate')
        plt.ylabel('% of apps approved')
        plt.xlabel('Date')
        
        plotnum +=1
    
    
    fig = plt.gcf()
    plt.show()







##################################################################
# Create a dataframe with the SSA dataset

print("Loading data")
ssa_df = initialize_data()


##################################################################
# Take the dataframe and reorganize such that the we end up with
# Date, Region, State, application count, and benefit type as feature columns

org_ssa_df = organize_data(ssa_df)


##################################################################
# Plot graphs to view the rate of application approvals for
# regions, states within region, and benefit types for state

print("\n\n")
print("plot allowance rate for all regions")
plot_regions(ssa_df, ['ATL', 'BOS', 'CHI', 'DAL', 'KCM', 'NYC', 'PHL', 'SEA', 'SFO'])

print("\n\n")
print("plot allowance rates for all states associated with PHL region")
plot_region(ssa_df, 'PHL', ['DC','DE','VA','PA','MD','WV'])

print("\n\n")
print("plot the allowance rate for each benefit type in VA state")
plot_types(ssa_df, 'PHL', 'VA')




##################################################################
# plot the distribution of allowance rate to see any outliers that may exist

print("\n\n")
print("Plot distribution of AllInitialAllowanceRate before removing outliers")
plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(ssa_df['AllInitialAllowanceRate'].min(), ssa_df['AllInitialAllowanceRate'].max()*1.1)
 
ssa_df['AllInitialAllowanceRate'].plot(kind='kde')
 
plt.subplot(212)
plt.xlim(ssa_df['AllInitialAllowanceRate'].min(), ssa_df['AllInitialAllowanceRate'].max()*1.1)
sns.boxplot(x=ssa_df['AllInitialAllowanceRate'])
plt.show()



##################################################################
# remove any outliers that have a z-score >1.8

print("\n\n")
print("remove outliers to smooth out the distribution")
ssa_df_filtered = pd.DataFrame(ssa_df[(np.abs(stats.zscore(ssa_df['AllInitialAllowanceRate'])) <=1.8)])




##################################################################
# now that outliers have been removed,
# plot the distribution of allowance rate to see what outliers may still exist

print("\n\n")
print("Plot distribution of AllInitialAllowanceRate after removing outliers")
plt.figure(figsize=(10,8))
plt.subplot(211)
plt.xlim(ssa_df_filtered['AllInitialAllowanceRate'].min(), ssa_df_filtered['AllInitialAllowanceRate'].max()*1.1)
 
ssa_df_filtered['AllInitialAllowanceRate'].plot(kind='kde')
 
plt.subplot(212)
plt.xlim(ssa_df_filtered['AllInitialAllowanceRate'].min(), ssa_df_filtered['AllInitialAllowanceRate'].max()*1.1)
sns.boxplot(x=ssa_df_filtered['AllInitialAllowanceRate'])
plt.show()





##################################################################
# now that outliers have been removed,
# plot graphs to view the rate of application approvals for
# regions, states within region, and benefit types for state

print("\n\n")
print("plot allowance rate for all regions")
plot_regions(ssa_df_filtered, ['ATL', 'BOS', 'CHI', 'DAL', 'KCM', 'NYC', 'PHL', 'SEA', 'SFO'])

print("\n\n")
print("plot allowance rates for all states associated with PHL region")
plot_region(ssa_df_filtered, 'PHL', ['DC','DE','VA','PA','MD','WV'])

print("\n\n")
print("plot the allowance rate for each benefit type in VA state")
plot_types(ssa_df_filtered, 'PHL', 'VA')









###############################################################################
# perform the same outlier removal on the re-organized dataset
# remove outliers in the re-organized dataset
ssa_df_filtered = pd.DataFrame(org_ssa_df[(np.abs(stats.zscore(org_ssa_df['InitialAllowanceRate'])) <=1.8)])



###############################################################################
# construct dataframe of binned targets
# the target values range from ~27 through ~47.  Therefore we want to create bins
# heavily based on these values.
conditions = [
    (ssa_df_filtered['InitialAllowanceRate'] >= 0) & (ssa_df_filtered['InitialAllowanceRate'] < 23),
    (ssa_df_filtered['InitialAllowanceRate'] >= 23) & (ssa_df_filtered['InitialAllowanceRate'] < 26),
    (ssa_df_filtered['InitialAllowanceRate'] >= 26) & (ssa_df_filtered['InitialAllowanceRate'] < 30),
    (ssa_df_filtered['InitialAllowanceRate'] >= 30) & (ssa_df_filtered['InitialAllowanceRate'] < 34),
    (ssa_df_filtered['InitialAllowanceRate'] >= 34) & (ssa_df_filtered['InitialAllowanceRate'] < 37),
    (ssa_df_filtered['InitialAllowanceRate'] >= 37) & (ssa_df_filtered['InitialAllowanceRate'] < 40),
    (ssa_df_filtered['InitialAllowanceRate'] >= 40) & (ssa_df_filtered['InitialAllowanceRate'] < 44),
    (ssa_df_filtered['InitialAllowanceRate'] >= 44) & (ssa_df_filtered['InitialAllowanceRate'] < 48),
    (ssa_df_filtered['InitialAllowanceRate'] >= 48) & (ssa_df_filtered['InitialAllowanceRate'] < 52),
    (ssa_df_filtered['InitialAllowanceRate'] >= 52)]
choices=[20,23,26,30,34,37,40,44,48,52]

ssa_df_filtered['InitialAllowanceRate_bin'] = np.select(conditions, choices, default=52)
ssa_allowance_rate_df = ssa_df_filtered[['InitialAllowanceRate_bin']]

# verify fairly equal distribution of these bins with:
# pd.value_counts(ssa_df_filtered['InitialAllowanceRate_bin'].values, sort=True)

del conditions
del choices


###############################################################################
# bin the InitialReceipts data into workload levels (Low, Medium, High)
# we'll use this rather than the precise numerical values
conditions = [
    (ssa_df_filtered['InitialReceipts'] >= 0) & (ssa_df_filtered['InitialReceipts'] < 420),
    (ssa_df_filtered['InitialReceipts'] >= 420) & (ssa_df_filtered['InitialReceipts'] < 1380),
    (ssa_df_filtered['InitialReceipts'] >= 1380)]
choices=['Low','Medium','High']

ssa_df_filtered['Workload'] = np.select(conditions, choices, default='High')

# verify fairly equal distribution of these bins with:
# pd.value_counts(ssa_df_filtered['Workload'].values, sort=True)

del conditions
del choices



###############################################################################
# bin the months into seasons (spring, summer, autumn, winter)
# we'll use seasonality rather than the precise month/date
conditions = [
        (ssa_df_filtered['Month'] == "01") | (ssa_df_filtered['Month'] == "02") | (ssa_df_filtered['Month'] == "12"),
        (ssa_df_filtered['Month'] == "03") | (ssa_df_filtered['Month'] == "04") | (ssa_df_filtered['Month'] == "05"),
        (ssa_df_filtered['Month'] == "06") | (ssa_df_filtered['Month'] == "07") | (ssa_df_filtered['Month'] == "08"),
        (ssa_df_filtered['Month'] == "09") | (ssa_df_filtered['Month'] == "10") | (ssa_df_filtered['Month'] == "11"),
        ]
choices=["Winter", "Spring", "Summer", "Autumn"]

ssa_df_filtered['Season'] = np.select(conditions, choices, default="Winter")

del conditions
del choices



###############################################################################
# specify the fields we plan to use for the x and y
x_df = ssa_df_filtered[['RegionCode','StateCode','Season','Workload','Type']]
y_df = ssa_df_filtered[['InitialAllowanceRate_bin']]


###############################################################################
# one-hot encode the factors in x_df
x_df = pd.get_dummies(x_df)



###############################################################################
# split data into test/train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_df, y_df, test_size=0.3, random_state=0)
#x_train.shape, y_train.shape
#x_test.shape, y_test.shape




###############################################################################
# Linear Regression

print("\n\n")
print("Testing with Linear Regression")

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)


# The coefficients
#print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error for Linear Regression: %.2f"
      % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction
print('Linear Regression Variance score: %.2f' % r2_score(y_test, y_pred))

del y_pred


###############################################################################
## SGD regressor

print("\n\n")
print("Testing with SGDRegressor")
clf = linear_model.SGDRegressor()

# Train using the training set
clf.fit(x_train, y_train)

# Make predictions with the test set
sgd_y_predicted = clf.predict(x_test)


# compare predicted to actual y_test values to compute measures of accuracy
# The mean squared error
print("Mean squared error for SGD: %.2f"
      % mean_squared_error(y_test, sgd_y_predicted))

# Explained variance score: 1 is perfect prediction
print('SGD Variance score: %.2f' % r2_score(y_test, sgd_y_predicted))

del sgd_y_predicted



###############################################################################
# Try KNN on the data
from sklearn.neighbors import KNeighborsClassifier

print("\n\n")
print("Testing with KNN")


#### Perform a loop across numerous options for value of K to determine most optimal selection
#
#### Store results
#train_accuracy = []
#test_accuracy  = []
#### Set KNN setting from 1 to 15
#knn_range = range(1, 15)
#for neighbors in knn_range:
#### Start Nearest Neighbors Classifier with K of 1
#  knn = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski', p=1)
#  ### Train the data using Nearest Neighbors
#  knn.fit(x_train, y_train)
#  ### Capture training accuracy
#  train_accuracy.append(knn.score(x_train, y_train))
#  ### Predict using the test dataset  
#  y_pred = knn.predict(x_test)
#  ### Capture test accuracy
#  test_accuracy.append(knn.score(x_test, y_test))
#  
### Plot Results from KNN Tuning
#plt.plot(knn_range, train_accuracy, label='training accuracy')
#plt.plot(knn_range, test_accuracy,  label='test accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Neighbors')
#plt.legend()
#plt.title('KNN Tuning ( # of Neighbors vs Accuracy')
#plt.savefig('KNNTuning.png')
#plt.show()
#
#del test_accuracy
#del train_accuracy


## KNN Using Output from KNN-Tuning, K = 9 is most ideal
knn = KNeighborsClassifier(n_neighbors=9,  metric='minkowski', p=1)

### Re-Train the data using Nearest Neighbors
knn.fit(x_train, y_train)

### Model Accuracy
y_pred= knn.predict(x_test)
print('\nPrediction from x_test:')
print(y_pred)

score = knn.score(x_test, y_test)
print('KNN score:', score*100,'%')


# The mean squared error
print("Mean squared error of KNN model: %.2f"
      % mean_squared_error(y_test, y_pred))

del y_pred
