import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression


# Read training set
sea_level_df = pd.read_csv("sealevel_train.csv")
sea_level_df.head()

# Read testing set
sea_level_df_test = pd.read_csv("sealevel_test.csv")
sea_level_df_test.head()


# Part a - Fit linear regression model to training data (find OLS coefficients) #

# Read training set

X = sea_level_df[['time']]
Y = sea_level_df[['level_variation']] 
print(type(X))



LR = LinearRegression().fit(X,Y)
LR.score(X, Y)

print("Coeffs:", LR.coef_)
print("Intercept:", LR.intercept_)


# Predict using OLS model

year_predict = 2013.453940

predict_val = year_predict*LR.coef_ + LR.intercept_
predict_vector = LR.coef_*X + LR.intercept_

print("The predicted value for 2013", predict_val)



# Plot training data along with the regression curve
plt.title ("Sea level over time training data")
plt.plot(X,Y)
plt.plot(X,predict_vector,'r') 




