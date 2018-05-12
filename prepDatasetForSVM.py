import sys
import os
import scipy.io.wavfile as wavfile
import numpy as np
from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures, getSegmentFeaturesUsingIndices
from sklearn.externals import joblib
import pandas as pd

def main():
    DIR = os.path.dirname(os.path.realpath(__file__))
    # load file for csv dataset
    # CSV_DATASET = DIR + "/datasetForSVM.csv"
    CSV_DATASET = DIR + "/datasetForSVM.csv"
    csv_dataset = open(CSV_DATASET, "w+")

    # load scaler
    scaler = joblib.load(DIR + "/scalerParameters.sav")
    print("Scaler loaded...")
    
    # load dnn
    dnn = joblib.load(DIR  + "/dnnParameters.sav")
    print("DNN loaded...")

    # get the dataset ready
    df = pd.read_csv(DIR + "/datasetForDNN.csv", header=None)
    print("Dataset loaded into dataframe...")

    # handle dataset
    # get dataframes of all emotions individually
    df0 = df.loc[df[326] == 0]
    df1 = df.loc[df[326] == 1]
    df2 = df.loc[df[326] == 2]
    df3 = df.loc[df[326] == 3]
   
    MIN_SAMPLES = np.min([df0.shape[0], df1.shape[0], df2.shape[0], df3.shape[0]])
    df0 = df0.iloc[0:MIN_SAMPLES,:]
    df1 = df1.iloc[0:MIN_SAMPLES,:]
    df2 = df2.iloc[0:MIN_SAMPLES,:]
    df3 = df3.iloc[0:MIN_SAMPLES,:]

    df = pd.concat([df0, df1, df2, df3])   
 
    # Split into X and y
    utteranceNames = df.iloc[:,0] # get utteranceNames
    utteranceNamesList = []
    for utteranceName in utteranceNames:
        if utteranceName not in utteranceNamesList:
            utteranceNamesList.append(utteranceName)
    
    X = df.iloc[:, 1:-1].values # remove the utteranceName and target emotionLabelNum
    y = df.iloc[:, -1].values # get target emotionLabelNum
    print("X and y loaded....")
   
    # normalize the data
    X = scaler.transform(X)
    print("Data normalized...")
    
    # calculate probabilities for all samples
    probabilities = dnn.predict_proba(X)
    print("Probabilities calculated...")
   
    print("Generating utterance feature vectors...")
    # setup progressBar
    progressBarWidth = 50
    sys.stdout.write("[%s]" % (" " * progressBarWidth))
    sys.stdout.flush()
    sys.stdout.write("\b" * (progressBarWidth+1)) # return to start of line, after '['
    sys.stdout.flush()

    progressBarUpdatePerUtterance = int(len(utteranceNamesList)/progressBarWidth)
    utteranceCount = 0

    df = df.set_index(0) # sets the utteranceNames as index
    segmentIndex = 0
    for utteranceName in utteranceNamesList:
    # utteranceName = utteranceNames[0]
        # print("UtteranceName : " + str(utteranceName))
        utteranceEmotionLabelNum = df.loc[utteranceName].iloc[0,-1] # emotion for the first segment of the utterance
        # print("Utterance emotion label : " + str(utteranceEmotionLabelNum))
        numberOfSegmentsPerUtterance = len(df.loc[utteranceName])
        # print("Number of segments per utterance : " + str(numberOfSegmentsPerUtterance))

        utteranceProbabilities = probabilities[segmentIndex:(segmentIndex+numberOfSegmentsPerUtterance),:]
        # print("Utterance probabilities : ")
        # print(utteranceProbabilities)

        # create the feature vector for utterance
        feat1 = np.amax(utteranceProbabilities, axis=0)
        feat2 = np.amin(utteranceProbabilities, axis=0)
        feat3 = np.mean(utteranceProbabilities, axis=0)

        prob0 = utteranceProbabilities[:,0]
        prob1 = utteranceProbabilities[:,1]
        prob2 = utteranceProbabilities[:,2]
        prob3 = utteranceProbabilities[:,3]
        # print("Prob 0 : ")
        # print(prob0)

        count0 = np.sum(prob0[prob0>0.2])/numberOfSegmentsPerUtterance
        count1 = np.sum(prob1[prob1>0.2])/numberOfSegmentsPerUtterance
        count2 = np.sum(prob1[prob2>0.2])/numberOfSegmentsPerUtterance
        count3 = np.sum(prob1[prob3>0.2])/numberOfSegmentsPerUtterance
        feat4 = np.array([count0, count1, count2, count3])

        featureVector = np.hstack([feat1, feat2, feat3, feat4])
        featureVectorString = ",".join(["%.8f" % num for num in featureVector])
        featureVectorString = utteranceName + "," + featureVectorString + "," + str(int(utteranceEmotionLabelNum))
        csv_dataset.write(featureVectorString + "\n")

        segmentIndex = numberOfSegmentsPerUtterance
        # print("New segment index")
        # print(segmentIndex)
        
        # update the progressBar
        utteranceCount += 1
        if (utteranceCount % progressBarUpdatePerUtterance == 0) and (int(utteranceCount/progressBarUpdatePerUtterance) <= 50): # won't let the progressbar #'s exceed 50 repetitions
            sys.stdout.write("#")
            sys.stdout.flush()

    sys.stdout.write("\n")
    csv_dataset.close()

if __name__ == '__main__':
    main()