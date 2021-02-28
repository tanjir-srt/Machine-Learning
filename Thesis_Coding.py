# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:08:17 2021

@author: Tanjir
"""
############################ import Library #########################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


########################### dataframe preparation ###################

df=pd.read_csv(r'C:\Users\Tanjir\Desktop\3G_Cell_Data_Check.csv')

#print(df.isnull().sum())

#print(df)



x=df[['Max HSDPA UE (None)','Mean HSDPA UE (None)','Cell Availability (%)']]
#print(x)


y=df[['HSDPA_Data_Volume_MB (MB)']]

#print(y)




###################### Data Split ##############################


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)


#print(len(x_train))
#print(y)
#print(x_test)
#print(y_test)

################### Regression model (Multiple variable linear regrassion) ######################


reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)

p=reg.predict(x_train)
y_pred=reg.predict(x_test)

#print(y_pred)


print(reg.score(x_test,y_test))

print(reg.predict([[80,60,100]]))

################## OLS ######################


est=sm.OLS(y_train,x_train).fit()

#print(est.summary())

############################### elbow method #############################


k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(x,y)
    sse.append(km.inertia_)
    
    
print(sse)    ### some of square error


plt.xlabel('K=number of cluster')

plt.ylabel('sse')

plt.plot(k_rng,sse)

############## K-means clustering ############


km=KMeans(n_clusters=3)
y_pred=km.fit_predict(x,y)

#print(y_pred)


df['cluster']=y_pred

#print(df)

df1=df[df.cluster==0]      ### all dataframe where cluster is zero
df2=df[df.cluster==1] 
df3=df[df.cluster==2]

#print(df1.mean())
#print(df2.mean())
#print(df3.mean())

################# ploting K-means clustering ##############

#plt.scatter(df1['Mean HSDPA UE (None)'],df1['HSDPA_Data_Volume_MB (MB)'],color='red')
#plt.scatter(df2['Mean HSDPA UE (None)'],df2['HSDPA_Data_Volume_MB (MB)'],color='blue')
#plt.scatter(df3['Mean HSDPA UE (None)'],df3['HSDPA_Data_Volume_MB (MB)'],color='Green')


#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',label='centroid',marker='+') # ploting centroid value
#plt.xlabel('Mean HSDPA UE')
#plt.ylabel('data Volume')
#plt.show()


############# Ploting 3D ##################

#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#ax.scatter(x_test['Mean HSDPA UE (None)'],x_test['Max HSDPA UE (None)'],y_test,c='r')
#ax.scatter(x_test['Mean HSDPA UE (None)'],x_test['Max HSDPA UE (None)'],y_pred,c='blue')
#ax.set_xlabel('Mean HSDPA UE (None)')
#ax.set_ylabel('Max HSDPA UE (None)')
#ax.set_zlabel('HSDPA_Data_Volume_MB (MB)')
#plt.show()
