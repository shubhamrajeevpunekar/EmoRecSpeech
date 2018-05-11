import glob
import os
import subprocess
import sys
import wave

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyaudio
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

    # load FILE
    try:
        FILE = sys.argv[1]
    except :
        FILE="keaton"


    MP4_DIR = os.path.join(DIR, "testMp4s")
    MP4_FILE = os.path.join(MP4_DIR, FILE+".mp4")

    print("Extracting audio from video : ", end= " ") # end=" " suppresses new line
    command = "ffmpeg -i " + MP4_FILE + " -ac 1 -ar 16000 -vn " + MP4_FILE[:-4] + ".wav"
    print(command)
    
    subprocess.call(command,shell=True)
    print("DONE")

    WAV_IN = os.path.join(MP4_DIR, FILE+".wav")
    testWav = wave.open(WAV_IN,"r")

   

    
    # aggregate 5 seconds of frames, process each 5 second utterance
    # NOTE : it is possible to directly read frames for 5 seconds 
    # i.e. (RATE*UTTERANCE_SECONDS), instead of reading them CHUNK by CHUNK 
    # and aggregating them, but we are using a for loop on CHUNKS, to 
    # keep it consistent with pyaudio stream input, which will be added later
    utteranceProbabilities = []
    
    utteranceCount = 0
    while(True):
        try:
            utterance = b'' # empty byte string
            for _ in range(int(RATE*UTTERANCE_SECONDS/CHUNK)):
                samples = testWav.readframes(CHUNK)
                utterance += samples
            
            # for testing, see if each 5 second utterance matches the deteced emotion
            WAV_OUT = os.path.join(MP4_DIR,FILE+ "_" +str(utteranceCount)+".wav")
            outWav = wave.open(WAV_OUT, "w")
            outWav.setnchannels(1)
            outWav.setsampwidth(2)
            outWav.setframerate(RATE)
            outWav.writeframes(utterance)
            outWav.close()
            print("Saved : " + str(WAV_OUT))
            
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
            utteranceProbabilities.append(avgSegmentProbabilities) # save for plotting
            print(MP4_FILE
     + " : " + str(utteranceCount) + " ---> " + emotions[emotionLabelNum])

            # update the utterance count, for the next CHUNKs read from the file
            
            utteranceCount += 1
        except :
            break
        
    utteranceProbabilities = np.array(utteranceProbabilities)    
    prob0 = utteranceProbabilities[:,0]
    prob1 = utteranceProbabilities[:,1]
    prob2 = utteranceProbabilities[:,2]
    prob3 = utteranceProbabilities[:,3]    
    # plot the data
    plt.xlabel("Seconds")  
    plt.ylabel("Confidence")  
    plt.plot(np.arange(1,(len(prob0))*UTTERANCE_SECONDS, UTTERANCE_SECONDS), prob0, label="neu", color="red")
    plt.plot(np.arange(1,(len(prob0))*UTTERANCE_SECONDS, UTTERANCE_SECONDS), prob1, label="sad_fea", color="blue")
    plt.plot(np.arange(1,(len(prob0))*UTTERANCE_SECONDS, UTTERANCE_SECONDS), prob2, label="ang_fru", color="green")
    plt.plot(np.arange(1,(len(prob0))*UTTERANCE_SECONDS, UTTERANCE_SECONDS), prob3, label="hap_exc_sur", color="black")
    plt.legend(loc="upper left")
    plt.show()
    

if __name__ == '__main__':
    main()
