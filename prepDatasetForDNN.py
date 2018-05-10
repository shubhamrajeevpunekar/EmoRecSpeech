import os
import sys
import glob
import numpy as np
import scipy.io.wavfile
from energy import getTopEnergySegmentsIndices
from featureExtraction import extractFeatures
from featureExtraction import getSegmentFeaturesUsingIndices

def main():
    # path to audio file dataset
    AUDIO_DATASET = os.path.dirname(os.path.realpath(__file__)) + "/dataset/"
    
    # path to feature extracted dataset for DNN
    # CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "/datasetForDNN.csv"
    CSV_DATASET = os.path.dirname(os.path.realpath(__file__)) + "/datasetForDNN_test.csv"

    # maintain a dictionary for emotion label and labelNumber
    emotions = {}
    emotionLabelNum = -1 # this will be needed to set the target in csv file

    emotionDirPaths = glob.glob(AUDIO_DATASET + "*")

    csv_dataset = open(CSV_DATASET, "w")

    for emotionDirPath in emotionDirPaths:
        # keep a count of segment per emotions
        countSegmentsPerEmotion = 0

        # emotionLabel is also the directory name in the dataset directory
        emotionLabel = emotionDirPath.split("/")[-1] 

        # set the emotion label
        emotions[emotionLabel] = emotionLabelNum
        emotionLabelNum += 1

        # for all files in the emotionLabel directory, generate csv data
        print("Generating csv data for : " + emotionLabel)

        wavFilesPath = os.path.join(AUDIO_DATASET, emotionLabel, "*")
        # print(wavFilesPath)

        # setup progressBar
        progressBarWidth = 50
        sys.stdout.write("[%s]" % (" " * progressBarWidth))
        sys.stdout.flush()
        sys.stdout.write("\b" * (progressBarWidth+1)) # return to start of line, after '['

        wavFiles = glob.glob(wavFilesPath)
        numberOfWavFiles = len(wavFiles)
        progressBarUpdatePerFiles = int(numberOfWavFiles/progressBarWidth)
        countFiles = 0
 
        for wavFile in wavFiles:
            # print(wavFile)
            audioRate, audioData = scipy.io.wavfile.read(wavFile)
            frameFeatureMatrix = extractFeatures(audioData, audioRate)
            topSegmentIndices = getTopEnergySegmentsIndices(audioData, audioRate)
            topSegmentFeatureMatrix = getSegmentFeaturesUsingIndices(frameFeatureMatrix, 25, topSegmentIndices)
            # for each top segment in audioData, write the feature vector into csv_dataset along with target emotionLabelNum
            for topSegmentIndex in range(len(topSegmentFeatureMatrix)):
                featureVector = ",".join(['%.8f' % num for num in topSegmentFeatureMatrix[topSegmentIndex]])
                featureVector = featureVector + "," + str(emotionLabelNum) + "\n"
                csv_dataset.write(featureVector)
                countSegmentsPerEmotion += 1

            # update the progressBar
            countFiles += 1
            if (countFiles % progressBarUpdatePerFiles == 0) and (int(countFiles/progressBarUpdatePerFiles) <= 50): # won't let the progressbar #'s exceed 50 repetitions
                sys.stdout.write("#")
                sys.stdout.flush()

        sys.stdout.write("\n")
        # print the count for each emotion
        print("Number of segments for emotion : " + emotionLabel + " [ " + str(emotionLabelNum) + " ] : " + str(countSegmentsPerEmotion))
    csv_dataset.close()

if __name__ == '__main__':
    main()