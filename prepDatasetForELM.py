import sys
import os
import scipy.io.wavfile as wavfile
import numpy as np
from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures, getSegmentFeaturesUsingIndices
from sklearn.externals import joblib
import pandas as pd

def main():
    # load file for csv dataset
    CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "datasetForELM.csv"
    # CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "datasetForELM_test.csv"
    csv_dataset = open(CSV_DATASET, "w+")

    # load scaler
    scaler = joblib.load("scalerParameters.sav")
    print("Scaler loaded...")
    
    # load dnn
    dnn = joblib.load("dnnParameters.sav")
    print("DNN loaded...")

    # get the dataset ready
    df = pd.read_csv("datasetForDNN.csv", header=None)
    print("Dataset loaded into dataframe...")
    
 
    # Split into X and y
    utteranceNames = df.iloc[:,0] # get utteranceNames
    X = df.iloc[:, 1:-1].values # remove the utteranceName and target emotionLabelNum
    y = df.iloc[:, -1].values # get target emotionLabelNum
    print("X and y loaded....")
   
    # normalize the data
    X = scaler.transform(X)
    print("Data normalized...")
    
    # calculate probabilities for all samples
    probabilities = dnn.predict_proba(X)
    print("Probabilities calculated...")

    df = df.set_index(0) # sets the utteranceNames as index

    segmentIndex = 0
    # for utteranceName in utteranceNames:
    utteranceName = utteranceNames[0]
    utteranceEmotionLabelNum = df.loc[utteranceName].iloc[0,-1] # emotion for the first segment of the utterance
    numberOfSegmentsPerUtterance = len(df.loc[utteranceName])

    utteranceProbabilities = probabilities[segmentIndex:(segmentIndex+numberOfSegmentsPerUtterance),:]
    # create the feature vector for utterance
    feat1 = np.amax(utteranceProbabilities, axis=0)
    feat2 = np.amin(utteranceProbabilities, axis=0)
    feat3 = np.mean(utteranceProbabilities, axis=0)

    prob0 = utteranceProbabilities[:,0]
    prob1 = utteranceProbabilities[:,1]
    prob2 = utteranceProbabilities[:,2]
    prob3 = utteranceProbabilities[:,3]

    count0 = np.sum(prob0[prob0>0.2])/numberOfSegmentsPerUtterance
    count1 = np.sum(prob1[prob1>0.2])/numberOfSegmentsPerUtterance
    count2 = np.sum(prob1[prob2>0.2])/numberOfSegmentsPerUtterance
    count3 = np.sum(prob1[prob3>0.2])/numberOfSegmentsPerUtterance
    feat4 = np.array([count0, count1, count2, count3])

    featureVector = np.hstack([feat1, feat2, feat3, feat4])

    print(featureVector)
    segmentIndex = numberOfSegmentsPerUtterance


    # # samples of same utterance
    # utteranceName = utteranceNames[0]   
    # utteranceEmotionLabelNum = df.loc[utteranceName].iloc[0,-1]
    # numberOfSegmentsPerUtterance = len(df.loc[utteranceName])

    # print("Utterance Name : "  + str(utteranceName))
    # print("Utterance EmotionLabelNum : " + str(utteranceEmotionLabelNum))
    # print("Number segments for utterance : " + str(numberOfSegmentsPerUtterance))


if __name__ == '__main__':
    main()