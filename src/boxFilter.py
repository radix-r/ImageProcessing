'''

Author: Ross Wagner

This program take 2 command line arguments. The path to an image to process
and the size of the filter to apply to it, either 3 or 5. The program applies a box filter
to the image and displays the result. resulting image saved in ../outputImages/boxFilter<filter size><original file name>


'''

from PIL import Image
import sys
import os


if sys.argv.__len__() < 3:
    print("Usage: python3 boxFilter.py <path to image> <size of filter. 3 or 5>")
    print("Example: python3 boxFilter.py ../inputImages/image1.png 5")
    exit(1)

size = int(sys.argv[2])
delta = (size-1)//2 # distance from center pixel to edge of filter

if size != 3 and size !=5:
    print("Filter size must be 3 or 5")
    exit(1)


im = Image.open(sys.argv[1])
pixelMap = im.load()


# get the file name for saving
filePath = sys.argv[1]
file = os.path.basename(filePath)

newImg = Image.new(im.mode, im.size)
newPixels = newImg.load()

for i in range(newImg.size[0]):
    # print()
    for j in range(newImg.size[1]):
        # print(pixelMap[i, j], end=" ")
        tempPix = 0
        k = i - delta
        while k <= i+delta:
            l = j - delta
            # l = -2
            while l <= j+delta:
                if k < 0 or k >= newImg.size[0] or l < 0 or l >= newImg.size[1]:
                    tempPix += 0
                else:
                    try:
                        tempPix += pixelMap[k, l]
                    except IndexError:
                        print("k: {0} |  l: {1}".format(k,l))
                        print("{0} | {1}".format(newImg.size[0], newImg.size[1]))
                        exit(1)

                l += 1
            k += 1

        tempPix = tempPix//(size*size)
        newPixels[i, j] = tempPix

newImg.show()
newImg.save("../outputImages/boxFilter{0}{1}".format(str(size), file), im.format)




