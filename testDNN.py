import sys
import os
import scipy.io.wavfile as wavfile
import numpy as np
from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures, getSegmentFeaturesUsingIndices
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn

def main():
    # load dnn params
    dnn = joblib.load(os.path.dirname(os.path.realpath(__file__)) + "/dnnParameters_2.sav")
    print("DNN loaded...")

    #  load test file
    testFile = "keaton.wav"
    testFilePath = os.path.dirname(os.path.realpath(__file__)) + "/testWavs/" + testFile
    audioRate, audioData = wavfile.read(testFilePath)
    frameFeatureMatrix = extractFeatures(audioData, audioRate)
    topSegmentIndices = getTopEnergySegmentsIndices(audioData, audioRate)
    topSegmentFeatureMatrix = getSegmentFeaturesUsingIndices(frameFeatureMatrix, 25, topSegmentIndices)
    print("Segments generated...")

    # normalize the data
    X_train = np.load("X_train.npy")
    scaler = StandardScaler()
    scaler.fit(X_train)
    print("Data normalized...")

    # for each segement generate the probability distribution
    segmentProbabilities = dnn.predict_proba(topSegmentFeatureMatrix)

    # convert to percent
    segmentProbabilities = segmentProbabilities * 100

    print(segmentProbabilities)

    # plot probability distribution
    prob0 = segmentProbabilities[:,0]
    prob1 = segmentProbabilities[:,1]
    prob2 = segmentProbabilities[:,2]
    prob3 = segmentProbabilities[:,3]

    # plot the data
    # plt.style.use('seaborn')
    plt.xlabel("Samples")  
    plt.ylabel("Confidence")  
    plt.plot(range(1,len(prob0)+1), prob0, label="neu", color="red")
    plt.plot(range(1,len(prob1)+1), prob1, label="sad_fea", color="blue")
    plt.plot(range(1,len(prob2)+1), prob2, label="ang_fru", color="green")
    plt.plot(range(1,len(prob3)+1), prob3, label="hap_exc_sur", color="black")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == '__main__':
    main()