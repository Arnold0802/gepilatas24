import cv2
import numpy as np

i=1
minimum_area = 100
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

print(len(contours))

approximated_contours = []
for contour in contours:
    if cv2.contourArea(contour) > minimum_area:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            approximated_contours.append(approx)

print(len(approximated_contours))

square_contours = [cnt for cnt in approximated_contours if len(cnt) == 4 and cv2.isContourConvex(cnt)]

print(len(square_contours))

img_height, img_width = bw_image.shape[:2]


#for contour in square_contours:
#    x, y, w, h = cv2.boundingRect(contour)
#    # Szűrés a kontúr arányainak alapján
#    if w / img_width < 0.8 and h / img_height < 0.8:
#        cv2.drawContours(bw_image, [contour], -1, (0, 255, 0), 2)


sorted_contours = sorted(square_contours, key=cv2.contourArea, reverse=True)
# Az első legnagyobb kizárása, ha az közel áll a kép méretéhez
for contour in sorted_contours[0:]:  # ha 1, az első elem kihagyása
    x, y, w, h = cv2.boundingRect(contour)
    # Itt végezheted el a további szűréseket és feldolgozást
    cv2.drawContours(bw_image, [contour], -1, (0, 255, 0), 2)



#largest_contour = max(square_contours, key=cv2.contourArea)
#x, y, w, h = cv2.boundingRect(largest_contour)
cropped_image = image[y:y+h, x:x+w]
cv2.imshow('Kivágott kép', cropped_image)
#cv2.imwrite('kivagott_kep.jpg', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


