# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 06:17:19 2021

@author: Tanjir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


df=pd.read_csv(r'C:\Users\Tanjir\Desktop\KPI_2G.csv')
#print(df)


plt.scatter(df[['HW Call Setup Success Rate (%)']],df['K3014:Traffic Volume on TCH (Erl)'])
#plt.show

x=df[['HW Call Setup Success Rate (%)','TR373:Cell Availability (%)','TCH Availability (%)']]
#print(x)

y=df[['K3014:Traffic Volume on TCH (Erl)']]

#print(y)
### Regression model

reg=linear_model.LinearRegression()
reg.fit(x,y)

p=reg.predict(x)

#print(p)
print(reg.predict([[100,100,100]]))
      

print('coef:',reg.coef_)
print('intercept:',reg.intercept_)

#plt.scatter(df[['age']],df['price'],c='red')
#plt.plot(x,pred,c='blue')
#plt.show()


