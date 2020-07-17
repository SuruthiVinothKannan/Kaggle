
#Initial Model Training with 1 fold
#export Training_data=Inputs/train_fold.csv
#export Fold=0
#export MODEL=$1
#python -m train

#Final Model Training with multiple folds
#export Training_data=Inputs/train_fold.csv
#export Test_data=Inputs/test.csv
#export MODEL=$1
#Fold=0 python -m train
#Fold=1 python -m train
#Fold=2 python -m train
#Fold=3 python -m train
#Fold=4 python -m train

# Final Model Prediction
export Training_data=Inputs/train_fold.csv
export Test_data=Inputs/test.csv
export MODEL=$1
python -m predict
