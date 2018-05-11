import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

# from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
# from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer

def main():
    # load dataset
    CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "/datasetForSVM.csv"
    df = pd.read_csv(CSV_DATASET, header=None)
    X = df.iloc[:,1:-1].values
    y = df.iloc[:,-1].values

    svm = SVC(probability=True, random_state=0,verbose=True)


    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    svm.fit(XTrain, yTrain)

    # dump
    svmParametersDump = os.path.dirname(os.path.realpath(__file__)) + "/svmParameters.sav"
    joblib.dump(svm, svmParametersDump)

    accuracy = svm.score(XTrain, yTrain)
    print("\n\n" + str(accuracy))

if __name__ == '__main__':
    main()