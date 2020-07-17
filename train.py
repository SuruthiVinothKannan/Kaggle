import os
import pandas as pd
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import metrics
import dispatcher
import joblib

Training_data = os.environ.get('Training_data')
Test_data = os.environ.get('Test_data')
Fold = int(os.environ.get('Fold'))
MODEL = os.environ.get('MODEL')

#Training_data = pd.read_csv('Inputs/train_fold.csv')
#Fold = 0

Fold_Mapping = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

if __name__=="__main__":
    df = pd.read_csv(Training_data)
    df_test = pd.read_csv(Test_data)
    train_df = df[df.kfold.isin(Fold_Mapping.get(Fold))]
    valid_df = df[df.kfold==Fold]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id","target","kfold"],axis=1)
    valid_df = valid_df.drop(["id","target","kfold"],axis=1)

    valid_df = valid_df[train_df.columns]

    label_encoders = []
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist() + df_test[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())
        label_encoders.append((c,lbl))

    #data is ready to train
    #clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs= -1, verbose=2)
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    #print(preds)
    print(metrics.roc_auc_score(yvalid,preds))

    joblib.dump(label_encoders, f"Models/{MODEL}_{Fold}_label_encoder.pkl")
    joblib.dump(clf, f"Models/{MODEL}_{Fold}.pkl")
    joblib.dump(train_df.columns, f"Models/{MODEL}_{Fold}_columns.pkl")

