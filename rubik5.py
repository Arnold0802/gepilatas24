import cv2
import numpy as np

i=5
minimum_area = 100
#for i in range (1,6):

imgneve = "minta"+str(i)+".png"
image = cv2.imread(imgneve)
#90%os fekete megtartása, minden más fehér - > nem jött be, most 75% -> majdnem jó, 50% .... nehéz olyat találni ami minden mintára megfelelő (itt már inverz)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
#bw_image = cv2.bitwise_not(thresholded_image)
#cv2.imshow('Csak 50% fekete', bw_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Megjelenítés
cv2.imshow('Fekete területek kijelölve', image)
cv2.waitKey(0)
cv2.destroyAllWindows()    
