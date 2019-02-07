'''

Author: Ross Wagner

This program take 3 command line arguments. The path to an image to process
and the size of the filter to apply to it, either 3, 5, or 7. The program applies a median filter
to the image and displays the result. resulting image saved in ../outputImages/medianFilter<filter size><original file name>


'''

from PIL import Image
import sys
import os
import array
import statistics


def main():
    if sys.argv.__len__() > 4 or sys.argv.__len__() < 2:
        print("Wrong number of command line arguments")
        print("For help: python3 pa1.py -help")
        exit(1)

    # check for action argument
    action = sys.argv[1]

    # verify input valid
    if action[0] == "-":
        action = action[1:]
    # take action according to input
    if action == "b" or action == "g" or action == "m":
        # get the file name for saving later
        filePath = sys.argv[2]
        file = os.path.basename(filePath)
        image = Image.open(filePath)
        mode = image.mode

        bands = image.split()
        filteredBands = []
        newImage = None

        size = int(sys.argv[3])
        # check for ../outputImages directory
        if not os.path.isdir("../outputImages"):
            os.makedirs("../outputImages")

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

        elif action == "g":
            print("Apply Gaussian size {0} on {1}".format(size, filePath))
            for band in bands:
                filteredBands.append(gaussianFilter(band, size))
            newImage = Image.merge(mode, filteredBands)
            # newImage.save("../outputImages/gaussianFilter{0}{1}".format(size, file), image.format)
        elif action == "m":
            print("Applying median filter size {0} on {1}".format(size, filePath))
            for band in bands:
                filteredBands.append(medianFilter(band, size))
            newImage = Image.merge(mode, filteredBands)
            newImage.save("../outputImages/medianFilter{0}{1}".format(size, file), image.format)
        newImage.show()

    elif action == "c":
        print("Applying Cannny edge detection")

    elif action == "h":
        print("Generating histogram")
    elif action == "help":
        printHelp()
        exit(0)
    else:
        print("Unrecognized command: {0}".format(action))
        print("For help: python3 pa1.py -help")


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


def gaussianFilter(image, size):
    return image


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
    print("Usage: python3 pa1.py -<action to take> <path to image>  [<size of filter. An odd number>]")
    print("Example: python3 pa1.py -m ../inputImages/image1.png  5")
    print("Apply image processing algorithms to apply filters, generate histograms, or apply Canny edge detection.\n")
    print("Potential mandatory arguments:")
    print("\t-b\t apply box filter. Size input needed.")
    print("\t-c\t apply Canny edge detection.")

    print("\t-g\t apply Gaussian filter. Size input needed.")
    print("\t-h\t generate a histogram of the image's color frequency.")

    print("\t-help\t print help.")

    print("\t-m\t apply median filter. Size input needed.")



if __name__ == "__main__":
    main()