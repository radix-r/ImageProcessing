'''

Author: Ross Wagner

This program take 3 command line arguments. The path to an image to process
and the size of the filter to apply to it, either 3, 5, or 7. The program applies a median filter
to the image and displays the result. resulting image saved in ../outputImages/medianFilter<filter size><original file name>


'''

from PIL import Image
import sys
import os
import statistics
import numpy as np
from scipy.stats import multivariate_normal
import math
import cv2


def main():
    numArgs = sys.argv.__len__()
    if numArgs > 5 or numArgs < 2:
        print("Wrong number of command line arguments")
        print("For help: python3 pa1.py -help")
        exit(1)

    # check for action argument
    action = sys.argv[1]

    # verify input valid
    if action[0] == "-":
        action = action[1:]




    # take action according to input

    if action == "help":
        printHelp()
        exit(0)


    filePath = sys.argv[2]
    # get the file name for saving later
    file = os.path.basename(filePath)
    image = Image.open(filePath)
    mode = image.mode

    bands = image.split()
    filteredBands = []
    newImage = None

    # check for ../outputImages directory
    if not os.path.isdir("../outputImages"):
        os.makedirs("../outputImages")

    # actions that only need the image
    if action == "gr":
        print("Generating gradient on {0}".format(filePath))

        ''' color stuff
        for band in bands:

            filteredBands.append(gradientForward(band))

        # restore the alpha channel
        if mode == "RGBA":
            filteredBands[3] = bands[3]

        newImage = Image.merge(mode, filteredBands)
        '''
        # convert image to grey scale
        newImage = gradientForward(image.convert('L'))
        print(type(mode))
        # newImage.save("../outputImages/gaussianFilter{0}-{1}{2}".format(size, sigma, file), image.format)

    elif action == "c":
        print("Applying Canny edge detection on {0}".format(filePath))
        # canny(image)

    elif action == "h":
        print("Generating histogram on {0}".format(filePath))
        # histogram(image)
        exit(0)

    elif action == "s":
        print("Applying Sobel edge detection on {0}".format(filePath))
        # canny(image)

    # actions that need image and size
    else:
        # get size input and make sure it is valid
        try:
            size = int(sys.argv[3])
        except ValueError:
            print("Size input must be a base 10 integer")
            print("For help: python3 pa1.py -help")
            exit(1)

        if size % 2 != 1:
            print("Filter size must be odd (3,5,7,...ect)")
            print("For help: python3 pa1.py -help")
            exit(1)

        if action == "b":
            print("Applying box filter size {0} on {1}".format(size, filePath))
            for band in bands:
                filteredBands.append(boxFilter(band, size))
            newImage = Image.merge(mode, filteredBands)
            newImage.save("../outputImages/boxFilter{0}{1}".format(size, file), image.format)

        elif action == "ga":
            # get sigma
            if numArgs == 5:
                try:
                    sigma = float(sys.argv[4])
                except ValueError:
                    print("Sigma input must be a base 10 float")
                    print("For help: python3 pa1.py -help")
                    exit(1)
            else:
                print("Sigma argument needed")
                print("For help: python3 pa1.py -help")
                exit(1)

            print("Applying Gaussian filter size {0} with sigma {1} on {2}".format(size, sigma, filePath))
            for band in bands:
                filteredBands.append(gaussianFilter2D(band, size, sigma))
            newImage = Image.merge(mode, filteredBands)
            newImage.save("../outputImages/gaussianFilter{0}-{1}{2}".format(size, sigma, file), image.format)

        elif action == "m":
            print("Applying median filter size {0} on {1}".format(size, filePath))
            for band in bands:
                filteredBands.append(medianFilter(band, size))
            newImage = Image.merge(mode, filteredBands)
            newImage.save("../outputImages/medianFilter{0}{1}".format(size, file), image.format)

        else:
            print("Unrecognized action: {0}".format(action))
            print("For help: python3 pa1.py -help")
            exit(1)

    if newImage is not None:
        newImage.show()



def boxFilter(image,size):

    pixelMap = image.load()
    newImg = Image.new(image.mode, image.size)
    newPixels = newImg.load()

    delta = (size - 1) // 2  # distance from center pixel to edge of filter

    # iterate through each pixel
    for i in range(newImg.size[0]):
        for j in range(newImg.size[1]):
            tempPix = 0
            k = i - delta
            # apply filter
            while k <= i + delta:
                l = j - delta
                while l <= j + delta:
                    if k < 0 or k >= newImg.size[0] or l < 0 or l >= newImg.size[1]:
                        tempPix += 0
                    else:
                        try:
                            tempPix += pixelMap[k, l]
                        except IndexError:
                            print("k: {0} |  l: {1}".format(k, l))
                            print("{0} | {1}".format(newImg.size[0], newImg.size[1]))
                            exit(1)

                    l += 1
                k += 1

            tempPix = tempPix // (size * size)
            newPixels[i, j] = tempPix
    return newImg

def gaussian(sigma, value):
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(value*value)/(2*sigma*sigma))


'''
ToDo
'''
def gaussianFilter2D(image, size, sigma):
    # iterate through each pixel
    delta = size // 2  # distance from center pixel to edge of filter

    a = np.linspace(0, size, size, endpoint=False)
    #b = multivariate_normal.pdf(a, mean=0, cov=sigma)
    for val in range(size):
        a[val] = gaussian(sigma, val-delta)
    b = a.reshape(1, size)

    gaussFilter = np.dot(b.T, b)
    #print(gaussFilter)

    # get the sum of the filter
    normalizer = sum2DMatrix(gaussFilter)


    pixelMap = image.load()
    newImg = Image.new(image.mode, image.size)
    newPixels = newImg.load()


    #newImg = cv2.filter2D(image,-1,gaussFilter)

    for x in range(newImg.size[0]):
        for y in range(newImg.size[1]):
            # apply filter
            tempPix = 0.0
            for filterX in range(size):

                for filterY in range(size):
                    # make sure filter is in range
                    targetX = x-delta+filterX
                    targetY = y-delta+filterY
                    if not (targetX < 0 or targetX >= newImg.size[0] or targetY < 0 or targetY >= newImg.size[1]):

                        try:
                            tempPix += float(gaussFilter[filterX][filterY]) * float(pixelMap[targetX, targetY])
                        except IndexError:
                            print("k: {0} |  l: {1}".format(filterX, filterY))
                            print("{0} | {1}".format(newImg.size[0], newImg.size[1]))
                            exit(1)

            newPixels[x, y] = int(tempPix/normalizer)

    return newImg


'''
Calculates forward gradient of a single band image using formula: G f(x)=f(x+1)-f(x)

'''
def gradientForward(image):
    pixelMap = image.load()
    gx = Image.new(image.mode, image.size)
    gy = Image.new(image.mode, image.size)
    gm = Image.new(image.mode, image.size)
    newPixelsX = gx.load()
    newPixelsY = gy.load()
    newPixelsMag = gm.load()

    width = image.size[0]
    height = image.size[1]

    # iterate through each pixel
    for x in range(width):
        for y in range(height):
            gradX = 0
            gradY = 0
            if x < width-1:
                gradX = pixelMap[x+1,y]-pixelMap[x,y]

            if y < height-1:
                gradY = pixelMap[x,y+1]-pixelMap[x,y]

            newPixelsX[x,y] = gradX
            newPixelsY[x,y] = gradY
            newPixelsMag[x,y] = int(math.sqrt(gradX*gradX + gradY*gradY))

    #gx.show()
    #gy.show()
    #gm.show()
    return gm



def medianFilter(image, size):
    # iterate through each pixel
    delta = (size - 1) // 2  # distance from center pixel to edge of filter
    # pixel count of filter. For median array
    #pixInFilter = size * size

    pixelMap = image.load()
    newImg= Image.new(image.mode, image.size)
    newPixels = newImg.load()

    for x in range(newImg.size[0]):
        for y in range(newImg.size[1]):

            filterX = x - delta
            # tempPixels = array.array('i', (0 for i in range(0,pixInFilter+1)))
            tempPixels = []
            # apply filter
            while filterX <= x + delta:
                filterY = y - delta
                while filterY <= y + delta:
                    if filterX < 0 or filterX >= newImg.size[0] or filterY < 0 or filterY >= newImg.size[1]:
                        tempPixels.append(0)
                    else:
                        try:
                            tempPixels.append(pixelMap[filterX, filterY])
                        except IndexError:
                            print("k: {0} |  l: {1}".format(filterX, filterY))
                            print("{0} | {1}".format(newImg.size[0], newImg.size[1]))
                            exit(1)

                    filterY += 1
                filterX += 1

            newPixels[x, y] = statistics.median(tempPixels)
    return newImg

def printHelp():
    print("Usage: python3 pa1.py -<action> <path to image> [<size of filter. An odd number>] [<sigma for Gauss>]")
    print("Example: python3 pa1.py -m ../inputImages/image1.png  5")
    print("Apply image processing algorithms to apply filters, generate histograms, or apply Canny edge detection.\n")
    print("Potential mandatory arguments:")
    print("\t-b\t apply box filter. Size input needed.")
    print("\t-c\t apply Canny edge detection.")

    print("\t-ga\t apply Gaussian filter. Size input needed. Sigma input needed")
    print("\t-gr\t get gradient.")
    print("\t-h\t generate a histogram of the image's color frequency.")

    print("\t-help\t print help.")

    print("\t-m\t apply median filter. Size input needed.")
    print("\t-s\t apply 3x3 Sobel filter.")


def sum2DMatrix(matrix):
    sum = 0.0
    for row in matrix:
        for elem in row:
            sum += elem
    return sum




if __name__ == "__main__":
    main()