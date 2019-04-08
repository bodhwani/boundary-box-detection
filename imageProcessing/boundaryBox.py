import cv2
from PIL import Image

from PIL import ImageFilter

import numpy

image = cv2.imread("images/1466511808550PictorStamp.png")

# cv2.imshow("blur",blur)
# cv2.waitKey(3000)
# cv2.destroyAllWindows()

noise = cv2.fastNlMeansDenoisingColored(image,None,17,21,10,10)

gray = cv2.cvtColor(noise,cv2.COLOR_BGR2GRAY) # grayscale
cv2.imshow("grey",gray)
cv2.waitKey(3000)
cv2.destroyAllWindows()
# blur = cv2.GaussianBlur(gray, (5, 5), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

thresh = cv2.Canny(gray, 150,255)
cv2.imshow("thresh image",thresh)
cv2.waitKey(6000)
cv2.destroyAllWindows()

# _,thresh = cv2.threshold(thresh,170,250,cv2.MORPH_CROSS)


# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# closing = cv2.bitwise_not(closing)
# cv2.imshow("c image",closing)
# cv2.waitKey(6000)
# cv2.destroyAllWindows()

gradient = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("g image",gradient)
cv2.waitKey(6000)
cv2.destroyAllWindows()        
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

dilated = cv2.dilate(gradient,kernel,iterations = 15)
erosion = cv2.erode(dilated, kernel, iterations=13)


# erosion = cv2.erode(dilated, kernel, iterations=10)
# gradient = cv2.morphologyEx(erosion, cv2.MORPH_GRADIENT, kernel)
cv2.imshow("e image",erosion)
cv2.waitKey(6000)
cv2.destroyAllWindows()

# opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# closing = cv2.bitwise_not(closing)



# kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))



# gray = erosion
contours, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) 

# contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(len(contours))

maxA=0
maxI=0
for a in range(len(contours)):
    currA=cv2.contourArea(contours[a])
    if(currA>maxA):
        maxA=currA
        maxI=a
(x,y,w,h) = cv2.boundingRect(contours[maxI])
im=cv2.rectangle(erosion,(x,y),(x+w,y+h),(255,255,255),6)

# acc = get_iou([x,y,x+w,y+h], [ac_x,ac_y,ac_x2,ac_y2])

cv2.imshow("IMAGE",im)
cv2.waitKey(6000)
cv2.destroyAllWindows()
# write original image with added contours to disk  
 
cv2.imwrite("contoured.jpg", im) 

