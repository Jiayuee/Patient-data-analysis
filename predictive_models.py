import os
import pandas as pd
import numpy as np
import xlsxwriter
from pandas import DataFrame

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

### Build predictive model for improvement in BP
# input: age, gender, baseline_bp, dyslipidaemia, prediabetes
# simple logistic regression

# prepare train data
ip = "edited_data.xlsx"
train_data = pd.read_excel(ip,sheet_name="BP Improvement")

cols = ['Age','Gender_Female','Gender_Male','Baseline Diastolic','Baseline Systolic',
        'Dyslipidaemia','Pre-Diabetes']
x = train_data[cols]
y = train_data['Diastolic Improvement']

model1 = LinearRegression(copy_X=True)
model2 = RandomForestRegressor(max_depth=3, random_state=0)
model3 = XGBRegressor()
model4 = LGBMRegressor()
models = [model1, model2, model3, model4]
model_names = []
scores2 = {}
for m in models:
    model_names.append(m.__class__.__name__)
    scores2[m.__class__.__name__] = {}
# there are two methods to cross validate. First method only gives you there
# the score, but does not give you a trained model. It is used for understanding
# the problem - which predictors and which models work better
# sklearn provides a function cross_val_score for this
scores1 = {} # scores1 is scores by method 1
for i in range(len(models)):
    scores1[i] = cross_val_score(models[i], x, y,
                                scoring='neg_mean_squared_error', cv=5)

for i in range(len(models)):
    print('{}: {}'.format(models[i].__class__.__name__, scores1[i]))

# method 2 involves going through train and test splis one by one
# sklearn provides a class KFold to help with that
kf = KFold(n_splits=5, shuffle=True)

for n, (train_index, test_index) in enumerate(kf.split(x)):
    ## we will make n models, one for each fold
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    for i in range(len(models)):
        model = models[i]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        this_score = mean_squared_error(y_test, y_pred)
        scores2[model_names[i]][n] = this_score
        # models_collection[model_names[i]].append(model)

scores2 = pd.DataFrame(scores2)

## now we have n models which can be used for future data,
## and average of outputs of these models can be used as the prediction
