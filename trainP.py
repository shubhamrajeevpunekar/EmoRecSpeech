from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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
    
    X = df.iloc[:,1:-1].values
    y = df.iloc[:,-1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    dnn = MLPClassifier(hidden_layer_sizes=(16), activation='relu',
                        solver='adam', max_iter=100, verbose=True,
                        early_stopping=True, validation_fraction=0.1)

    dnn.fit(X_train, y_train)
    y_pred = dnn.predict(X_test)

    print("misclassified samples : %d" % (y_test!=y_pred).sum())
    print("total samples " + str(len(y_test)))
if __name__ == '__main__':
    main()