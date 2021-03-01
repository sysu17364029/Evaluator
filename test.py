import argparse
import glob
import os
import shutil
import sys

import _init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from Evaluator import *
from utils import BBFormat
import pv

def test(gtFileName,detFileName):

# Get current path to set default folders
#currentPath = os.path.dirname(os.path.abspath(__file__))
    gtFile = gtFileName
    detFile = detFileName

    iouThreshold = 0.5

    # Validate formats
    gtFormat = BBFormat.XYWH
    detFormat = BBFormat.XYWH

    # Coordinates types
    gtCoordType = CoordinatesType.Absolute
    detCoordType = CoordinatesType.Absolute
    imgSize = (0, 0)

    # Get groundtruth boxes
    allBoundingBoxes, allClasses = pv.getBoundingBoxes(
        gtFile, True, gtFormat, gtCoordType, imgSize=imgSize)

    # Get detected boxes
    allBoundingBoxes, allClasses = pv.getBoundingBoxes(
        detFile, False, detFormat, detCoordType, allBoundingBoxes, allClasses, imgSize=imgSize)
    allClasses.sort()

    evaluator = Evaluator()
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
        allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
        IOUThreshold=iouThreshold,  # IOU threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=False,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
        savePath=None,
        showGraphic=False)

    # each detection is a class
    for metricsPerClass in detections:

        # Get metric values per each class
        cl = metricsPerClass['class']
        ap = metricsPerClass['AP']
        precision = metricsPerClass['precision']
        recall = metricsPerClass['recall']
        totalPositives = metricsPerClass['total positives']
        total_TP = metricsPerClass['total TP']
        total_FP = metricsPerClass['total FP']

        if totalPositives > 0:
            validClasses = validClasses + 1
            acc_AP = acc_AP + ap
            prec = ['%.2f' % p for p in precision]
            rec = ['%.2f' % r for r in recall]
            ap_str = "{0:.2f}%".format(ap * 100)
            # ap_str = "{0:.4f}%".format(ap * 100)
            print('AP: %s (%s)' % (ap_str, cl))

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    return mAP_str

def main():
    test('gt','dr')

if __name__ == '__main__':
    main()
