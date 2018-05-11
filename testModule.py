import glob
import os

import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from sklearn.externals import joblib

from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures, getSegmentFeaturesUsingIndices


def main():
    emotions = ['neu','sad_fea', 'ang_fru','hap_exc_sur']

    DIR = os.path.dirname(os.path.realpath(__file__))
 
    # load wav
    WAVS_DIR = os.path.join(DIR, "testWavs")
    testWavs = glob.glob(WAVS_DIR + "/*")
    # print(testWavs)

    for i in range(len(testWavs)):
        # detect features
        audioRate, audioData = wavfile.read(testWavs[i])    
        frameFeatureMatrix = extractFeatures(audioData, audioRate)
        topSegmentIndices = getTopEnergySegmentsIndices(audioData, audioRate)
        topSegmentFeatureMatrix = getSegmentFeaturesUsingIndices(frameFeatureMatrix, 25, topSegmentIndices)

        # normalize data
        scaler = joblib.load(DIR + "/scalerParameters.sav")
        topSegmentFeatureMatrix = scaler.transform(topSegmentFeatureMatrix)

        # generate probabilities with DNN
        dnn = joblib.load(DIR  + "/dnnParameters.sav")
        segmentProbabilities = dnn.predict_proba(topSegmentFeatureMatrix)

        # create high level features
        feat1 = np.amax(segmentProbabilities, axis=0)

        feat2 = np.amin(segmentProbabilities, axis=0)
        feat3 = np.mean(segmentProbabilities, axis=0)
        prob0 = segmentProbabilities[:,0]
        prob1 = segmentProbabilities[:,1]
        prob2 = segmentProbabilities[:,2]
        prob3 = segmentProbabilities[:,3]
        count0 = np.sum(prob0[prob0>0.5])/len(segmentProbabilities)
        count1 = np.sum(prob1[prob1>0.5])/len(segmentProbabilities)
        count2 = np.sum(prob1[prob2>0.5])/len(segmentProbabilities)
        count3 = np.sum(prob1[prob3>0.5])/len(segmentProbabilities)
        feat4 = np.array([count0, count1, count2, count3])

        featureVector = np.hstack([feat1, feat2, feat3, feat4])

        # predict with svm
        # svm = joblib.load(DIR + "/svmParameters.sav")
        # emotionLabelNum, = svm.predict(featureVector.reshape(1,-1))

        emotionLabelNum = np.argmax(feat3)
        # display result
        print(testWavs[i].split("/")[-1][:-4] + " ---> " + emotions[emotionLabelNum])


if __name__ == '__main__':
    main()
