from skimage.measure import compare_ssim
import imutils
import warnings 
import cv2
import numpy as np

warnings.filterwarnings(action = 'ignore')

def resizeImages(image):
    dim = (1000, 1000)
    return cv2.resize(image, dim)

def imgToGray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculateMse(imageA, imageB):
    err= np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

animals = ["dog-1.png", "dog-2.png", "dog-3.png", "elephant-1.jpeg", "elephant-2.png", "elephant-3.jpeg"]

imageA = cv2.imread("animals/{}".format(animals[0]))
imageA = resizeImages(imageA)
imageA = imgToGray(imageA)
for animal in animals:
    imageB = cv2.imread("animals/{}".format(animal))
    imageB = resizeImages(imageB)
    imageB = imgToGray(imageB)
    (score, diff) = compare_ssim(imageA, imageB, full=True)
    mse = calculateMse(imageA, imageB)
    print("For {} & {}, SSIM:  {}, MSE: {}".format(animals[0], animal, score, mse))
