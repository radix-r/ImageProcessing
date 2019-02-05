'''

Author: Ross Wagner

This program take 2 command line arguments. The path to an image to process
and the size of the filter to apply to it, either 3, 5, or 7. The program applies a median filter
to the image and displays the result. resulting image saved in ../outputImages/medianFilter<filter size><original file name>


'''

from PIL import Image
import sys
import os
import array
import statistics


if sys.argv.__len__() < 3:
    print("Usage: python3 medianFilter.py <path to image> <size of filter. 3 or 5>")
    print("Example: python3 boxFilter.py ../inputImages/image1.png 5")
    exit(1)

size = int(sys.argv[2])
delta = (size-1)//2 # distance from center pixel to edge of filter
# pixel count of filter. For median array
pixInFilter = size*size

if size % 2 != 1:
    print("Filter size must be odd (3,5,7,...ect)")
    exit(1)


im = Image.open(sys.argv[1])
pixelMap = im.load()


# get the file name for saving later
filePath = sys.argv[1]
file = os.path.basename(filePath)

newImg = Image.new(im.mode, im.size)
newPixels = newImg.load()

# iterate through each pixel
for x in range(newImg.size[0]):
    for y in range(newImg.size[1]):

        filterX = x - delta
        # tempPixels = array.array('i', (0 for i in range(0,pixInFilter+1)))
        tempPixels = []
        tempPixIndex = 0
        # apply filter
        while filterX <= x+delta:
            filterY = y - delta
            while filterY <= y+delta:
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
            tempPixIndex +=1

        newPixels[x, y] = statistics.median(tempPixels)

newImg.show()
newImg.save("../outputImages/medianFilter{0}{1}".format(size, file), im.format)




