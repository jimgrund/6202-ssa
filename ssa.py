#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 11:15:33 2018

@author: jimgrund
"""

import os
### Provide the path here
os.chdir('/Users/jimgrund/Documents/GWU/machine learning/midterm_project/') 

### Basic Packages
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import scipy.stats as stats

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import datetime
import matplotlib.dates as mdates
import seaborn as sns
import gc

from sklearn import cross_validation


## DataLoad and Global Filtering

### read in the SSA data
#ssa_df  = pd.read_csv('SSA-SA-MOWL.csv', index_col=False, names=['Filename', 'z', 'FileDate', 'RegionCode', 'StateCode', 'FileLocation', 'FullDate', 'FormattedDate', 'AllInitialReceipts', 'AllInitialClosingPending', 'AllInitialDeterminations', 'AllInitialAllowances', 'AllInitialAllowanceRate', 'SSDIReceipts', 'SSDIClosingPending', 'SSDIDeterminations', 'SSDIAllowances', 'SSDIAllowanceRate', 'SSIReceipts', 'SSIClosingPending', 'SSIDeterminations', 'SSIAllowances', 'SSIAllowanceRate', 'ConcReceipts', 'ConcClosingPending', 'ConcDeterminations', 'ConcAllowances', 'ConcAllowanceRate', 'SSIDisabledChildReceipts', 'SSIDisabledChildClosingPending', 'SSIDisabledChildDeterminations', 'SSIDisabledChildAllowances', 'SSIDisabledChildAllowanceRate', 'PrototypeState', 'AllReconsReceipts', 'AllReconsClosingPending', 'AllReconsDeterminations', 'AllReconsAllowances', 'AllReconsAllowanceRate', 'SSDIReconsReceipts', 'SSDIReconsClosingPending', 'SSDIReconsDeterminations', 'SSDIReconsAllowances', 'SSDIReconsAllowanceRate', 'SSIReconsReceipts', 'SSIReconsClosingPending', 'SSIReconsDeterminations', 'SSIReconsAllowances', 'SSIReconsAllowanceRate', 'ConcReconsReceipts', 'ConcReconsClosingPending', 'ConcReconsDeterminations', 'ConcReconsAllowances', 'ConcReconsAllowanceRate', 'SSIDisabledChildReconsReceipts', 'SSIDisabledChildReconsClosingPending', 'SSIDisabledChildReconsDeterminations', 'SSIDisabledChildReconsAllowances', 'SSIDisabledChildReconsAllowanceRate', 'AllCDRReceipts', 'AllCDRClosingPending', 'AllCDRDeterminations', 'AllCDRAllowances', 'AllCDRAllowanceRate', 'SSDICDRReceipts', 'SSDICDRClosingPending', 'SSDICDRDeterminations', 'SSDICDRAllowances', 'SSDICDRAllowanceRate', 'SSICDRReceipts', 'SSICDRClosingPending', 'SSICDRDeterminations', 'SSICDRAllowances', 'SSICDRAllowanceRate', 'ConcCDRReceipts', 'ConcCDRClosingPending', 'ConcCDRDeterminations', 'ConcCDRAllowances', 'ConcCDRAllowanceRate','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n'])

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


#x1 = ssa_df.query('RegionCode == "ATL" and StateCode == "AL"')['FullDate']
#y1 = ssa_df.query('RegionCode == "ATL" and StateCode == "AL"')['AllInitialReceipts']

#x1 = ssa_df.query('RegionCode == "ATL"').groupby(['FullDate'], as_index=False)['FullDate']




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
        
        # Plot
        #fig = plt.figure()
        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')
    
        # Now append the extra data
        #x1.append(date_end)
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
    
    
    
    
    #fig.autofmt_xdate() # This tidies up the x axis
    fig = plt.gcf()
    plt.show()



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
        
        # Plot
        #fig = plt.figure()
        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')
    
        # Now append the extra data
        #x1.append(date_end)
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
    
    
    
    #fig.autofmt_xdate() # This tidies up the x axis
    fig = plt.gcf()
    plt.show()



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
        
        # Plot
        #fig = plt.figure()
        fig.add_subplot(330+plotnum)
    
        # Plot actual data
        plt.plot_date(x=x1, y=y1, fmt='o-')
    
        # Now append the extra data
        #x1.append(date_end)
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
    
    
    
    #fig.autofmt_xdate() # This tidies up the x axis
    fig = plt.gcf()
    plt.show()



print("Loading data")
ssa_df = initialize_data()

org_ssa_df = organize_data(ssa_df)

print("\n\n")
print("plot allowance rate for all regions")
plot_regions(ssa_df, ['ATL', 'BOS', 'CHI', 'DAL', 'KCM', 'NYC', 'PHL', 'SEA', 'SFO'])

print("\n\n")
print("plot allowance rates for all states associated with PHL region")
plot_region(ssa_df, 'PHL', ['DC','DE','VA','PA','MD','WV'])

print("\n\n")
print("plot the allowance rate for each benefit type in VA state")
plot_types(ssa_df, 'PHL', 'VA')







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





print("\n\n")
print("remove outliers to smooth out the distribution")
#cc_df_filtered = ssa_df[(np.abs(stats.zscore(ssa_df['AllInitialAllowanceRate'])) <=0.08)]
ssa_df_filtered = pd.DataFrame(ssa_df[(np.abs(stats.zscore(ssa_df['AllInitialAllowanceRate'])) <=1.8)])





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








print("\n\n")
print("plot allowance rate for all regions")
plot_regions(ssa_df_filtered, ['ATL', 'BOS', 'CHI', 'DAL', 'KCM', 'NYC', 'PHL', 'SEA', 'SFO'])

print("\n\n")
print("plot allowance rates for all states associated with PHL region")
plot_region(ssa_df_filtered, 'PHL', ['DC','DE','VA','PA','MD','WV'])

print("\n\n")
print("plot the allowance rate for each benefit type in VA state")
plot_types(ssa_df_filtered, 'PHL', 'VA')








# org_ssa_df
# remove outliers in the re-organized dataset
ssa_df_filtered = pd.DataFrame(org_ssa_df[(np.abs(stats.zscore(org_ssa_df['InitialAllowanceRate'])) <=1.8)])




# construct df of binned targets
conditions = [
    (ssa_df_filtered['InitialAllowanceRate'] >= 0) & (ssa_df_filtered['InitialAllowanceRate'] < 28),
    (ssa_df_filtered['InitialAllowanceRate'] >= 28) & (ssa_df_filtered['InitialAllowanceRate'] < 30),
    (ssa_df_filtered['InitialAllowanceRate'] >= 30) & (ssa_df_filtered['InitialAllowanceRate'] < 32),
    (ssa_df_filtered['InitialAllowanceRate'] >= 32) & (ssa_df_filtered['InitialAllowanceRate'] < 34),
    (ssa_df_filtered['InitialAllowanceRate'] >= 34) & (ssa_df_filtered['InitialAllowanceRate'] < 36),
    (ssa_df_filtered['InitialAllowanceRate'] >= 36) & (ssa_df_filtered['InitialAllowanceRate'] < 38),
    (ssa_df_filtered['InitialAllowanceRate'] >= 38) & (ssa_df_filtered['InitialAllowanceRate'] < 40),
    (ssa_df_filtered['InitialAllowanceRate'] >= 40) & (ssa_df_filtered['InitialAllowanceRate'] < 42),
    (ssa_df_filtered['InitialAllowanceRate'] >= 42) & (ssa_df_filtered['InitialAllowanceRate'] < 44),
    (ssa_df_filtered['InitialAllowanceRate'] >= 44) & (ssa_df_filtered['InitialAllowanceRate'] < 46),
    (ssa_df_filtered['InitialAllowanceRate'] >= 46)]
choices=[0,1,2,3,4,5,6,7,8,9,10]

ssa_df_filtered['InitialAllowanceRate_bin'] = np.select(conditions, choices, default=10)
ssa_allowance_rate_df = ssa_df_filtered[['InitialAllowanceRate_bin']]


# bin the InitialReceipts data into workload levels (Low, Medium, High)
conditions = [
    (ssa_df_filtered['InitialReceipts'] >= 0) & (ssa_df_filtered['InitialReceipts'] < 5000),
    (ssa_df_filtered['InitialReceipts'] >= 5000) & (ssa_df_filtered['InitialReceipts'] < 10000),
    (ssa_df_filtered['InitialReceipts'] >= 10000)]
choices=['Low','Medium','High']

ssa_df_filtered['Workload'] = np.select(conditions, choices, default='High')




x_df = ssa_df_filtered[['RegionCode','StateCode','Month','Workload','Type']]
y_df = ssa_df_filtered[['InitialAllowanceRate_bin']]

#from sklearn.preprocessing import OneHotEncoder
x_df = pd.get_dummies(x_df)



#####
# split data into test/train
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_df, y_df, test_size=0.3, random_state=0)
x_train.shape, y_train.shape
x_test.shape, y_test.shape



####
# Logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2', max_iter=1000)
model.fit(x_train,y_train)
prediction=model.predict(x_test)



#### 
# Try KNN on the data
from sklearn.neighbors import KNeighborsClassifier


### Store results
train_accuracy = []
test_accuracy  = []
### Set KNN setting from 1 to 15
knn2_range = range(1, 15)
for neighbors in knn2_range:
### Start Nearest Neighbors Classifier with K of 1
  knn2 = KNeighborsClassifier(n_neighbors=neighbors,
                              metric='minkowski', p=1)
### Train the data using Nearest Neighbors
  knn2.fit(x_train, y_train)
### Capture training accuracy
  train_accuracy.append(knn2.score(x_train, y_train))
### Predict using the test dataset  
  y_pred = knn2.predict(x_test)
### Capture test accuracy
  test_accuracy.append(knn2.score(x_test, y_test))
  
## Plot Results from KNN Tuning
plt.plot(knn2_range, train_accuracy, label='training accuracy')
plt.plot(knn2_range, test_accuracy,  label='test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Neighbors')
plt.legend()
plt.title('KNN Tuning ( # of Neighbors vs Accuracy')
plt.savefig('KNNTuning.png')
plt.show()
