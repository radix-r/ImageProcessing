'''

Author: Ross Wagner

This program take 2 command line arguments. The path to an image to process
and the size of the filter to apply to it, either 3 or 5. The program applies a box filter
to the image and displays the result


'''

from PIL import Image
import sys


if sys.argv.__len__() < 3:
    print("Usage: python3 boxFilter.py <path to image> <size of filter. 3 or 5>")
    print("Example: python3 boxFilter.py ../inputImages/image1.png 5")
    exit(1)


size = int(sys.argv[2])

if size != 3 and size !=5:
    print("Filter size must be 3 or 5")
    exit(1)


im = Image.open(sys.argv[1])
pixelMap = im.load()

newImg = Image.new(im.mode, im.size)
newPixels = newImg.load()

for i in range(newImg.size[0]):
    for j in range(newImg.size[1]):
        newPixels[i, j] = pixelMap[i, j]//2


newImg.show()




