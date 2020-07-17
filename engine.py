import pandas as pd
import os

#final = os.environ.get('Models/randomforest.csv')
df = pd.read_csv('Models/randomforest.csv')

print(df.dtypes)