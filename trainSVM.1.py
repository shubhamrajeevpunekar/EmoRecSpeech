import pandas as pd
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn_extensions.extreme_learning_machines.elm import GenELMClassifier
# from sklearn_extensions.extreme_learning_machines.random_layer import MLPRandomLayer

def main():
    # load dataset
    CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "/datasetForSVM.csv"
    df = pd.read_csv(CSV_DATASET, header=None)
    
    # HANDLING IMBALANCE

    # get dataframes of all individually
    df0 = df.loc[df[17] == 0]
    df1 = df.loc[df[17] == 1]
    df2 = df.loc[df[17] == 2]
    df3 = df.loc[df[17] == 3]
   
    # take 1000 samples of each
    df0 = df0.iloc[0:1000,:]
    df1 = df1.iloc[0:1000,:]
    df2 = df2.iloc[0:1000,:]
    df3 = df3.iloc[0:1000,:]

    df = pd.concat([df0,df1,df2,df3])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # print(df)

    X1 = df.iloc[:,1:5]
    X2 = df.iloc[:,9:13]
    X = np.hstack([X1,X2])
    y = df.iloc[:,-1].values



   
    svm = SVC(kernel='rbf',probability=True, random_state=0,verbose=True)


    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=0)
    
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)
    
    svm.fit(XTrain, yTrain)

    # dump
    svmParametersDump = os.path.dirname(os.path.realpath(__file__)) + "/svmParameters.sav"
    joblib.dump(svm, svmParametersDump)

    accuracy = svm.score(XTest, yTest)
    print("\n\n" + str(accuracy))

    y_pred = svm.predict(XTest)
    print("wrong output : " + str(sum(y_pred!=yTest)))
    print("Total : " + str(len(y_pred)))
if __name__ == '__main__':
    main()