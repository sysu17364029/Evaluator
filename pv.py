import argparse
import glob
import os
import shutil
# from argparse import RawTextHelpFormatter
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat


def getBoundingBoxes(directory,
              isGT,
              bbFormat,
              coordType,
              allBoundingBoxes=None,
              allClasses=None,
              imgSize=(0, 0)):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    if allBoundingBoxes is None:
        allBoundingBoxes = BoundingBoxes()
    if allClasses is None:
        allClasses = []
        
    # Read ground truths
    #os.chdir(directory)
    #files = glob.glob("*.txt")
    f = directory+".txt"
    
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box
    # (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box

    nameOfImage = f.replace(".txt", "")
    fh1 = open(f, "r")
    for line in fh1:
        line = line.replace("\n", "")
        if line.replace(' ', '') == '':
            continue
        splitLine = line.split(" ")
        if isGT:
            name0fImage = "0001"
            # idClass = int(splitLine[0]) #class
            idClass = (splitLine[0])  # class
            #print(idClass)
            x = float(splitLine[1])
            #print("x:",x)
            y = float(splitLine[2])
            #print("y:",y)
            w = float(splitLine[3])
            #print("w:",w)
            h = float(splitLine[4])
            #print("h:",h)
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.GroundTruth,
                format=bbFormat)
        else:
            name0fImage = "0001"
            # idClass = int(splitLine[0]) #class
            idClass = (splitLine[0])  # class
            #print(idClass)
            confidence = float(splitLine[1])
            x = float(splitLine[2])
            #print("x:",x)
            y = float(splitLine[3])
            #print("y:",y)
            w = float(splitLine[4])
            #print("w:",w)
            h = float(splitLine[5])
            #print("h:",h)
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                coordType,
                imgSize,
                BBType.Detected,
                confidence,
                format=bbFormat)
        allBoundingBoxes.addBoundingBox(bb)
        if idClass not in allClasses:
            allClasses.append(idClass)
    fh1.close()
    print("allBoundingBoxes:",allBoundingBoxes)
    print("allClasses",allClasses)
    return allBoundingBoxes, allClasses
