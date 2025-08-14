from calib import *
ref=cv.imread('test_data/ref.jpg')
rm = Remove_marker(ref, True, False)
cv.imshow('bg',rm.bgr)
cv.waitKey(0)