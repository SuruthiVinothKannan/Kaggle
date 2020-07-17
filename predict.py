import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import dispatcher
import joblib
import numpy as np


Test_data = os.environ.get('Test_data')
MODEL = os.environ.get('MODEL')

def predict():
    df = pd.read_csv(Test_data)
    test_id = df["id"].values
    predictions = None
    
    for Fold in range(5):
        print(Fold)
        df = pd.read_csv(Test_data)
        encoders = joblib.load(os.path.join("Models", f"{MODEL}_{Fold}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("Models", f"{MODEL}_{Fold}_columns.pkl"))
        for (c,lbl) in encoders:
            print(c)
            #lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        #data is ready to train
        clf = joblib.load(os.path.join("Models", f"{MODEL}_{Fold}.pkl"))
    
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if Fold == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_id, predictions)), columns=["id","target"])
    return sub    

if __name__=="__main__":
    submission = predict()
    submission.id = submission.id.astype(int)
    submission.to_csv(f"Models/{MODEL}.csv", index=False)