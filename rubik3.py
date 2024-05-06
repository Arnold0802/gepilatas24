import cv2
import numpy as np

image = cv2.imread('minta1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # A legnagyobb kontúr keresése
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)  # Zöld színű keretet rajzol



if hierarchy is not None:
    # Végigmegyünk minden kontúron
    for i in range(len(contours)):
        if hierarchy[0][i][3] == 0:  # Gyermek kontúrok a legnagyobb kontúron belül
            cv2.drawContours(image, [contours[i]], -1, (255, 0, 0), 2)  # Piros színű keretet rajzol

print(len(contours))


cv2.imshow('Kiemelt négyzetek', image)
#cv2.imwrite('kiemelt_négyzetek.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
