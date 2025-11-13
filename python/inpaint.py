#!/usr/bin/env -S uvx autopep723

import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True)
parser.add_argument("-m", "--mask", type=str, required=True)
parser.add_argument("-o", "--output", type=str, required=True)
args = parser.parse_args()

image = cv.imread(args.image)
mask = cv.imread(args.mask, cv.IMREAD_GRAYSCALE)
final = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)
cv.imwrite(args.output, final)
