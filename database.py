import os
import numpy as np
import cv2
import pickle
from segmenter import segmentary, readImageAndResize

def createDataBase():
    originalImagesControl = os.listdir("origcontrol")
    originalImagesPatient = os.listdir("origpatient")

    dataset = []

    for fileName in originalImagesControl:
        filePath = "./origcontrol/" + fileName

        image = readImageAndResize(filePath)
        blueLineImage, _ = segmentary(image)
        blueLineImage[blueLineImage == 255] = 1

        dataset.append((blueLineImage, 0))

    for fileName in originalImagesPatient:
        filePath = "./origpatient/" + fileName

        image = readImageAndResize(filePath)
        blueLineImage, _ = segmentary(image)
        blueLineImage[blueLineImage == 255] = 1

        dataset.append((blueLineImage, 1))
    
    dataset = np.array(dataset)
    np.random.shuffle(dataset)

    with open("./_dataset.pd", "wb") as f:
        pickle.dump(dataset, f)

def readDataBaseImage():
    with open("./_dataset.pd", "rb") as f:
        dataset = pickle.load(f)
    
    return np.array(dataset[:,0]), np.array(dataset[:,1])

