#this is the most basic model
#importing the dataset
import pandas as pd

data = pd.read_csv('Advertising.csv')
print(data.isna().sum())

X =data[['TV', 'Radio', 'Newspaper']].values
#X will have the values of the features

#normalizing the target variable
X = (X - X.mean(axis=0)) / X.std(axis=0)

X = X.astype('float32')


y = data['Sales'].values.reshape(-1,1)
y = y.astype('float32')
