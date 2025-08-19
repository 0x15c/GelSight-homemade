from calib import *
ref=cv.imread('test_data_new/ref.jpg')
rm = Remove_marker(ref, True, True)
cv.imshow('bg',rm.bgr)
cv.imwrite("bg.jpg",rm.bgr)
cv.waitKey(0)