This is best viewed with a markdown renderer to view embedded images: https://atom.io/packages/markdown-preview

# Overview
Usage: python3 pa1.py -\<action> \<path to image> \[<size of filter. An odd number>] \[<sigma for Gauss>]

Example: python3 pa1.py -m ../inputImages/image1.png  5

Apply image processing algorithms to apply filters, generate histograms, or apply Canny edge detection.

Potential mandatory arguments:

        -b       apply box filter. Size input needed.
        -c       apply Canny edge detection. Take high and low inputs
        -e       apply max entropy threshold.
        -ga      apply Gaussian filter. Size input needed. Sigma input needed
        -ga2     apply Gaussian filter using slower 2d method. Size input needed. Sigma input needed
        -gr      get gradient. Defaults to centralized method
        -grb     get gradient using backward method
        -grf     get gradient using forward method 
        -h       generate a histogram of the image's intensity and frequency.
        -help    print help.
        -m       apply median filter. Size input needed.
        -s       apply 3x3 Sobel filter.
