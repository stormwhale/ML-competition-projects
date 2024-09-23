from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd

#import sample file
data = pd.read_csv('https://raw.githubusercontent.com/stormwhale/ML-competition-projects/refs/heads/main/melb_data.csv')

#define X and y
y = data.Price
predictor= data.drop(['Price'], axis=1)

##only using numerical data:
X = predictor.select_dtypes(exclude=['object'])

#split data:
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0)

#check for missing data in num_col:
mia_col = [col for col in X_train.columns if X_train[col].isnull().any()]
mia_col
X_train[mia_col].isnull().sum()/13580*100 #determine the precentage of missing data.

#pipline for dropping columns that have >30% data and imputing for 'Car' column
def drop(Z):
    return Z.drop(['BuildingArea','YearBuilt'], axis = 1)

model = RandomForestRegressor(n_estimators=100, random_state=0)
my_pipe = Pipeline([
    ('drop', FunctionTransformer(drop, validate=False)),
    ('im', SimpleImputer()),
    ('model', model)
])

#fitting model
my_pipe.fit(X_train, y_train)
predict = my_pipe.predict(X_valid)

#calculate MAE
score = mean_absolute_error(y_valid, predict)
print(score)

#comparing MAE without dropping the two columns:
my_pipe2 = Pipeline([
    ('im', SimpleImputer()),
    ('model', model)
])

#fitting model
my_pipe2.fit(X_train, y_train)
predict = my_pipe2.predict(X_valid)

score2 = mean_absolute_error(y_valid, predict)
print(score2)