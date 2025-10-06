from joblib import load
import datetime
import os

model = load('ml/models/model_VALE3.SA.joblib')
dt = datetime.date.today()
print('Predict for', dt)
print(model.predict([[dt.toordinal()]]))
