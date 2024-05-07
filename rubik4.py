import cv2
import numpy as np

i=1
#for i in range (1,6):

imgneve = "minta"+str(i)+".png"
image = cv2.imread(imgneve)
#90%os fekete megtartása, minden más fehér - > nem jött be, most 75% -> majdnem jó, 50% .... nehéz olyat találni ami minden mintára megfelelő
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
bw_image = cv2.bitwise_not(thresholded_image)
cv2.imshow('Csak 50% fekete', bw_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


contours, hierarchy = cv2.findContours(bw_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

approximated_contours = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approximated_contours.append(approx)

square_contours = [cnt for cnt in approximated_contours if len(cnt) == 4 and cv2.isContourConvex(cnt)]
for contour in square_contours:
    cv2.drawContours(bw_image, [contour], -1, (0, 255, 0), 2)


largest_contour = max(square_contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

cropped_image = bw_image[y:y+h, x:x+w]

cv2.imshow('Kivágott kép', cropped_image)
#cv2.imwrite('kivagott_kep.jpg', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

