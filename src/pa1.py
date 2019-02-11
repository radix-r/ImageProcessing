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
import math
import matplotlib.pyplot as plt
import time


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



    # start timer
    start = time.time()

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
    if action.startswith("gr"):

        version = action[-1:]
        grade = None

        if version == "f":
            print("Generating forward gradient on {0}".format(filePath))
            # convert image to grey scale
            grade = gradient(image.convert('L'), "f")
            newImage = grade["m"]

        elif version == "b":
            print("Generating backward gradient on {0}".format(filePath))
            # convert image to grey scale
            grade = gradient(image.convert('L'), "b")
            newImage = grade["m"]

        else: # default to central
            version = "c"
            print("Generating central gradient on {0}".format(filePath))
            # convert image to grey scale
            grade = gradient(image.convert('L'), "c")
            newImage = grade["m"]

        grade["x"].show()
        grade["y"].show()
        newImage.save("../outputImages/gradient-{0}-{1}".format(version, file), newImage.format)
        ''' color stuff
        for band in bands:

            filteredBands.append(gradientForward(band))

        # restore the alpha channel
        if mode == "RGBA":
            filteredBands[3] = bands[3]

        newImage = Image.merge(mode, filteredBands)
        '''

    elif action == "c":
        # get low and high values from cmd line
        if numArgs == 5:
            try:
                high = int(sys.argv[3])
                low = int(sys.argv[4])
            except ValueError:
                print("High/Low input must be a base 10 int")
                print("For help: python3 pa1.py -help")
                exit(1)
        else:
            print("High/low argument needed")
            print("For help: python3 pa1.py -help")
            exit(1)

        print("Applying Canny edge detection on {0}".format(filePath))
        newImage = cannyEdgeDetection(image.convert("L"), high, low)
        newImage.save("../outputImages/cannyEdgeDetection{0}-{1}-{2}".format(high,low,file), newImage.format)


    elif action == "e":
        print("Applying entropy threshold on {0}".format(filePath))
        newImage =entropyThreshold(image.convert("L"))
        newImage.save("../outputImages/entropyThreshold-{0}".format(file), newImage.format)

    elif action == "h":
        print("Generating histogram from {0}".format(filePath))
        values = histogram(image.convert("L"))
        plt.bar(range(256), values)
        plt.title("Histogram of Image {0}".format(filePath))
        plt.xlabel("Intensity")
        plt.ylabel("Frequency")
        plt.show()
        exit(0)

    elif action == "s":
        print("Applying Sobel edge detection on {0}".format(filePath))
        sob = sobelFilter(image.convert("L"))
        sob['x'].save("../outputImages/sobelFilter-x-{0}".format(file), sob['x'].format)
        sob['y'].save("../outputImages/sobelFilter-y-{0}".format(file), sob['y'].format)

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



        elif action.startswith("ga"):
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
                if action.endswith("2"):
                    filteredBands.append(gaussianFilter2D(band, size, sigma))
                else:
                    filteredBands.append(gaussianFilter1D(band, size, sigma))
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

    end = time.time()
    print("Operation took {0} seconds".format(end-start))

'''
applies threshold to grey scale image
'''
def applyThreshold(image, threshold):
    pixelMap = image.load()

    binary = Image.new(image.mode, image.size)
    binaryPix= binary.load()

    width =image.size[0]
    height = image.size[1]

    for x in range(width):
        for y in range(height):
            if pixelMap[x,y] >= threshold:
                binaryPix[x,y] = 255
            else:
                binaryPix[x,y] = 0

    return binary

'''
This function applies a simple box that sets each pixel to the average of the pixels around it  filter averaging 
'''
def boxFilter(image,size):

    pixelMap = image.load()
    newImg = Image.new(image.mode, image.size)
    newPixels = newImg.load()

    delta = (size - 1) // 2  # distance from center pixel to edge of filter

    # iterate through each pixel
    for x in range(newImg.size[0]):
        for y in range(newImg.size[1]):
            tempPix = 0
            k = x - delta
            # apply filter
            while k <= x + delta:
                l = y - delta
                while l <= y + delta:
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
            newPixels[x, y] = tempPix
    return newImg

'''
uses gaussian of size 5 with sigma 1
'''
def cannyEdgeDetection(image, high, low):
    # gaussian
    gauss = gaussianFilter1D(image,5,1)

    # gradient
    grad = gradient(gauss, "c")

    # non-max
    nMax = nonMax(grad["m"], grad["a"])

    # hysteresis

    hys = hysteresis(nMax, high, low)
    # best sigma for smoothing

    return hys



'''
This function take in a grey scale image and returns a binary image
with threshold creating max entropy. only seems effective with simple images
'''
def entropyThreshold(image):
    # get histogram
    hist = histogram(image)

    total = sum(hist)

    len = hist.__len__()

    indexOfMaxEntropy = 0
    maxEntropy = 0.0

    pt = 0.0
    # try each value T
    for T in range(len):
        # calculate entropy
        pt += hist[T]/total
        if pt == 0:
            continue
        a = 0.0
        b = 0.0

        for i in range(len):

            if hist[i] <= 0:
                continue
            if i <= T:
                pi = hist[i] / (total * pt)
                a += pi * math.log(pi, 10)
            else:

                pi = hist[i]/(total*(1-pt))

                b += pi * math.log(pi, 10)

            tempEntropy = math.fabs(a + b)
            if tempEntropy > maxEntropy:
                maxEntropy = tempEntropy
                indexOfMaxEntropy = T

    print("Max entropy at intensity threshold {0}".format(indexOfMaxEntropy))

    binary = applyThreshold(image, indexOfMaxEntropy)

    return binary


'''
'''
def gaussian(sigma, value):
    return (1/(math.sqrt(2*math.pi)*sigma))*math.exp(-(value*value)/(2*sigma*sigma))

'''
apply gaussian filter to image using 1d method. About 4 times faster than 2d method
using image3-1.png:
2d: 21.4 sec
1d: 5.3 sec
'''
def gaussianFilter1D(image,size, sigma):
    # iterate through each pixel
    delta = size // 2  # distance from center pixel to edge of filter

    a = np.linspace(0, size, size, endpoint=False)
    # b = multivariate_normal.pdf(a, mean=0, cov=sigma)
    for val in range(size):
        a[val] = gaussian(sigma, val - delta)
    gaussFilter = a
    # print(gaussFilter)
    normalizer = a.sum()

    width = image.size[0]
    height = image.size[1]

    pixelMap = image.load()
    yPass = Image.new(image.mode, image.size)
    yPixles = yPass.load()
    xPass = Image.new(image.mode, image.size)
    xPixles = xPass.load()

    # filter y direction
    for x in range(width):
        for y in range(height):
            tempPix = 0.0
            for index in range(size):
                offset = y - delta+index
                if offset >= 0 and offset < height:
                    tempPix += pixelMap[x,offset]*gaussFilter[index]

            yPixles[x,y]= int (tempPix/normalizer)

    # filter x direction
    for x in range(width):
        for y in range(height):
            tempPix = 0.0
            for index in range(size):
                offset = x - delta + index
                if offset >= 0 and offset < width:
                    tempPix += yPixles[offset, y] * gaussFilter[index]

            xPixles[x,y] = int(tempPix / normalizer)

    return xPass


'''
Generates a gaussian kernel based on size and sigma. very slow b/c of iteration through image and kernal
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

Generates backward gradient using equation: G f(x) = f(x)-f(x-1)

Generates central gradient using equation: G f(x) = f(x+1)-f(x-1)
'''
def gradient(image, version):
    pixelMap = image.load()
    gx = Image.new(image.mode, image.size)
    gy = Image.new(image.mode, image.size)
    gm = Image.new(image.mode, image.size)
    newPixelsX = gx.load()
    newPixelsY = gy.load()
    newPixelsMag = gm.load()

    width = image.size[0]
    height = image.size[1]

    angleMatrix = [[0 for x in range(height)] for y in range(width)]

    # iterate through each pixel
    for x in range(width):
        for y in range(height):
            gradX = 0
            gradY = 0

            if version == "f":
                if x < width - 1:
                    gradX = pixelMap[x + 1, y] - pixelMap[x, y]

                if y < height - 1:
                    gradY = pixelMap[x, y + 1] - pixelMap[x, y]
            elif version == "b":
                if x > 0:
                    gradX = pixelMap[x, y] - pixelMap[x - 1, y]

                if y > 0:
                    gradY = pixelMap[x, y] - pixelMap[x, y - 1]
            else:
                if x > 0 and x < (width - 1):
                    gradX = pixelMap[x + 1, y] - pixelMap[x - 1, y]

                if y > 0 and y < height - 1:
                    gradY = pixelMap[x, y + 1] - pixelMap[x, y - 1]

            newPixelsX[x, y] = gradX
            newPixelsY[x, y] = gradY
            newPixelsMag[x, y] = int(math.sqrt(gradX * gradX + gradY * gradY))
            if gradX == 0:
                if gradY ==0:
                    angleMatrix[x][y]=0
                else:
                    angleMatrix[x][y] = gradY/math.fabs(gradY) * math.pi/2
            else:
                try:
                    angleMatrix[x][y] = math.atan(gradY/gradX)
                except IndexError:
                    print("Out of bounds: {0} {1}".format(x,y))
                    exit(1)

    # gm.show()
    return {"x": gx, "y": gy, "m": gm, "a": angleMatrix}


'''
Counts the number of occurrences of each value in an 8 bit grey scale image
retruns a list size 256 where each intensity(index) is assigned its frequency 
'''
def histogram(image):
    values = [0]*256

    pixelMap = image.load()

    width = image.size[0]
    height = image.size[1]

    for x in range(width):
        for y in range(height):
            values[pixelMap[x,y]] += 1



    return values


def hysteresis(image, high,low):
    inPixles = image.load()

    out = Image.new("1", image.size)
    outPixles = out.load()

    width = image.size[0]
    height = image.size[1]

    for x in range(width):
        for y in range(height):
            # check if passes high pass
            if inPixles[x,y] >= high:
                outPixles[x,y] = 1
            elif inPixles[x,y] < low:
                outPixles[x,y] = 0
            else:
                # check neighboring pixels, look for at least 2 above low
                passCount = 0
                examineX = x-1
                for checkX in range(3):
                    examineY = y-1
                    for checkY in range(3):
                        # check that we are in range
                        if examineX < width and examineX >= 0 and examineY < height and examineY >= 0:
                            if inPixles[examineX,examineY] > low:
                                passCount += 1

                        examineY += 1

                    examineX += 1

                if passCount > 1:
                    outPixles[x, y] = 1

    return out
'''
'''
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


''''''
def nonMax(grad, angles):
    width = grad.size[0]
    height = grad.size[1]

    normalizer = 1/math.sqrt(2)

    gradPixles = grad.load()
    filtered = Image.new(grad.mode, grad.size)
    filteredPixels = filtered.load()

    for x in range(width):
        for y in range(height):
            xDir = math.cos(angles[x][y])/normalizer
            yDir = math.sin(angles[x][y])/normalizer

            xDir = round(xDir, 0)
            yDir = round(yDir, 0)

            keep = True
            # check positive direction
            examineX = x+xDir
            examineY = y+yDir

            # Make sure in bounds
            for i in range(2):
                if  examineX < width and examineX >= 0 and examineY < width and examineY >=0:
                    if gradPixles[examineX,examineY] > gradPixles[x,y]:
                        keep = False

                examineX = x - xDir
                examineY = y - yDir

            if keep:
                filteredPixels[x,y] = gradPixles[x,y]
            else:
                filteredPixels[x, y] = 0

    return filtered

def printHelp():
    print("Usage: python3 pa1.py -<action> <path to image> [<size of filter. An odd number>] [<sigma for Gauss>]")
    print("Example: python3 pa1.py -m ../inputImages/image1.png  5")
    print("Apply image processing algorithms to apply filters, generate histograms, or apply Canny edge detection.\n")
    print("Potential mandatory arguments:")
    print("\t-b\t apply box filter. Size input needed.")
    print("\t-c\t apply Canny edge detection. Take high and low inputs")
    print("\t-e\t apply max entropy threshold.")

    print("\t-ga\t apply Gaussian filter. Size input needed. Sigma input needed")
    print("\t-ga2\t apply Gaussian filter using slower 2d method. Size input needed. Sigma input needed")

    print("\t-gr\t get gradient. Defaults to centralized method")
    print("\t-grb\t get gradient using backward method")
    print("\t-grf\t get gradient using forward method ")
    print("\t-h\t generate a histogram of the image's intensity and frequency.")

    print("\t-help\t print help.")

    print("\t-m\t apply median filter. Size input needed.")
    print("\t-s\t apply 3x3 Sobel filter.")


'''
This function applies a 3x3 sobel filter to a given image and returns the result


'''
def sobelFilter(image):
    kernalX = [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]

    kernalY = [[-1, -2, -1],[0, 0, 0],[1, 2, 1]]


    delta = 1

    pixelMap = image.load()
    sobX = Image.new(image.mode, image.size)
    sobY = Image.new(image.mode, image.size)
    pixX = sobX.load()
    pixY = sobY.load()

    width = image.size[0]
    height = image.size[1]

    for x in range(width):
        for y in range(height):

            tempX=0
            tempY=0
            # apply filter
            for i in range(3):
                filterX = x - i - delta

                for j in range(3):
                    filterY = y - j - delta
                    # check if out of bounds
                    if not(filterX < 0 or filterX >= width or filterY < 0 or filterY >= height):
                        tempX += kernalX[i][j]*pixelMap[filterX, filterY]
                        tempY += kernalY[i][j] * pixelMap[filterX, filterY]
                    filterY += 1
                filterX += 1

            pixX[x, y] = tempX
            pixY[x, y] = tempY

    sobX.show()
    sobY.show()
    return {'x': sobX, 'y': sobY}


def sum2DMatrix(matrix):
    total = 0.0
    for row in matrix:
        for elem in row:
            total += elem
    return total


if __name__ == "__main__":
    main()
