import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_excel("C:/Users/sabarishmanogaran/OneDrive - revature.com/Desktop/DS/Project/Procurement Fraudness.xlsx")

df['Unit Price'].unique()

df['Unit Price'].value_counts()

df['InflatedInvoice'].value_counts()

df['InflatedInvoice'].unique()

df['Employees colluding with suppliers with higher cost'].value_counts()

df['Employees colluding with suppliers with higher cost'].unique()

df = df [['Unit Price', 'InflatedInvoice', 'Employees colluding with suppliers with higher cost', 'Fraudness']] 

from sklearn.preprocessing import LabelEncoder

# Creating instance of labelencoder
labelencoder = LabelEncoder()

y = df.iloc[:, 3:]

y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

df_new = pd.concat([df,y], axis =1)
df_new.drop(['Fraudness'], axis = 1, inplace = True)
df_new = df_new.rename(columns={0:'Fraudness'})

#Converting Float into int
df_new.dtypes
#df.dtypes
#df.InflatedInvoice = df.InflatedInvoice.astype('int64')

df_new.InflatedInvoice = df.InflatedInvoice.astype('int64')
df_new.dtypes

#df.to_csv('procurementfraudness.csv',encoding="utf-8")
df_new.to_csv('procurementfraudness.csv',encoding="utf-8")
import os
os.getcwd()

#Model Building
train, test = train_test_split(df, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, :3], train.iloc[:, 3])
#help(LogisticRegression)

test_predict = model.predict(test.iloc[:, :3]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:, 3], test_predict)

train_predict = model.predict(train.iloc[:, :3]) # Train predictions 
# Train accuracy 

accuracy_score(train.iloc[:, 3], train_predict)

#X = df.iloc[:, :3]
X = df_new.iloc[:, :3]
#y = df.iloc[:, 3]
y = df_new.iloc[:, 3]

regressor = LogisticRegression(multi_class = "multinomial", solver = "newton-cg")

#Fitting model with trainig data
regressor.fit(X, y)

#Making a Predictive System

import pickle

# Saving model to disk
filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))
#pickle.dump(regressor, open('model1.pkl','wb'))

# Loading model to compare the results
loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
#loaded_model = pickle.load(open('C:/Users/sabarishmanogaran/finalized_model.sav', 'rb'))
#model1 = pickle.load(open('model1.pkl','rb'))

input_data = (80,548,1499)

input_data_as_numpy_as_array = np.asarray(input_data)

#resahpe the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_as_array.reshape(1,-1)

Prediction = loaded_model.predict(input_data_reshaped)
print(Prediction)

if (Prediction[0] == 0):    
   print('Procurement Fraud does not happen')
else:
    print('Procurement Fraud happens')
