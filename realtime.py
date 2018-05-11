import glob
import os

import numpy as np
import pandas as pd
import pyaudio
import wave
from sklearn.externals import joblib

from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures, getSegmentFeaturesUsingIndices


def main():
    emotions = ['neu','sad_fea', 'ang_fru','hap_exc_sur']

    DIR = os.path.dirname(os.path.realpath(__file__))
 
    # default parameters
    RATE = 16000
    CHUNK = 1024
    DEVICE_IN_HW = "Camera"
    DEVICE_OP_HW = "pulse"
    UTTERANCE_SECONDS = 5

    # load wav
    WAVS_DIR = os.path.join(DIR, "testWavs")
    WAV_FILE = os.path.join(WAVS_DIR, "keaton.wav")
    testWav = wave.open(WAV_FILE,"r")

    
    WAV_OUT = "out.wav"
    outWav = wave.open(WAV_OUT, "w")
    outWav.setnchannels(1)
    outWav.setsampwidth(2)
    outWav.setframerate(RATE)

    utterance = b'' # empty byte string
    for _ in range(int(RATE*UTTERANCE_SECONDS/CHUNK)):
        samples = testWav.readframes(CHUNK)
        utterance += samples
    outWav.writeframes(utterance)
    outWav.close()

    utterance = np.fromstring(utterance, np.int16)
    frameFeatureMatrix = extractFeatures(utterance, RATE)
    topSegmentIndices = getTopEnergySegmentsIndices(utterance, RATE)
    topSegmentFeatureMatrix = getSegmentFeaturesUsingIndices(frameFeatureMatrix, 25, topSegmentIndices)

    # normalize data
    scaler = joblib.load(DIR + "/scalerParameters.sav")
    topSegmentFeatureMatrix = scaler.transform(topSegmentFeatureMatrix)

    # generate probabilities with DNN
    dnn = joblib.load(DIR  + "/dnnParameters.sav")
    segmentProbabilities = dnn.predict_proba(topSegmentFeatureMatrix)

    # create high level features
    avgSegmentProbabilities = np.mean(segmentProbabilities, axis=0)
    
    # determine emotionLabelNum
    emotionLabelNum = np.argmax(avgSegmentProbabilities)

    # display result
    print("Probabilities : " + str(avgSegmentProbabilities))
    print(WAV_FILE + " ---> " + emotions[emotionLabelNum])


if __name__ == '__main__':
    main()
