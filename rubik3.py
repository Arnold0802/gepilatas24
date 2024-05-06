import cv2
import numpy as np

def dominant_color(cell):
    # A kép átalakítása 2D-s tömbbé, ahol minden sor egy képpont
    data = np.reshape(cell, (-1, 3))
    
    # Leggyakoribb szín keresése
    colors, count = np.unique(data, axis=0, return_counts=True)
    dominant = colors[count.argmax()]
    return dominant

def equalize_histogram_color(image):
    # BGR-ből YUV-ra konvertálás
    yuv_img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Hisztogram egyenlítés a Y csatornán
    yuv_img[:,:,0] = cv2.equalizeHist(yuv_img[:,:,0])
    # Visszakonvertálás BGR-be
    equalized_img = cv2.cvtColor(yuv_img, cv2.COLOR_YUV2BGR)
    return equalized_img


image = cv2.imread('minta1.png')
image = equalize_histogram_color(image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Gaussian Blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Vagy Median Blur
#blurred = cv2.medianBlur(gray, 5)


#_, thresholded = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#if contours:
#    # A legnagyobb kontúr keresése
#    largest_contour = max(contours, key=cv2.contourArea)
#    cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 2)  # Zöld színű keretet rajzol
#
#
#
#if hierarchy is not None:
#    # Végigmegyünk minden kontúron
#    for i in range(len(contours)):
#        if hierarchy[0][i][3] == 0:  # Gyermek kontúrok a legnagyobb kontúron belül
#            cv2.drawContours(image, [contours[i]], -1, (255, 0, 0), 2)  # Piros színű keretet rajzol

print(len(contours))

approximated_contours = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approximated_contours.append(approx)

square_contours = [cnt for cnt in approximated_contours if len(cnt) == 4 and cv2.isContourConvex(cnt)]
for contour in square_contours:
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)



cv2.imshow('Kiemelt négyzetek', image)
#cv2.imwrite('kiemelt_négyzetek.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#megvan a nagy négyzet, egész jól körülhatárolva, most ezzel a képpel kéne tovább menni és megtalálni benne a kicsiket.

largest_contour = max(square_contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest_contour)

cropped_image = image[y:y+h, x:x+w]

cv2.imshow('Kivágott kép', cropped_image)
#cv2.imwrite('kivagott_kep.jpg', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#mivel megvan a nagy négyzet, amiben benne van a 9 kicsi, innentől 2 irányba indulhatok el. felosztom 9 egyenlő részre a nagy négyzetet, 
#majd megkeresem hogy adott négyzetben melyik a leggyakoribb szín, 
#vagy próbálkozom a 9 négyzet megtalálásával, majd ezekben keresem meg a leggyakoribb színt. az első opció egyszerűbbnek tűnik

height, width, _ = cropped_image.shape
cell_height = height // 3
cell_width = width // 3

# Kisebb négyzetek listájának létrehozása
cells = []
for i in range(3):  # Sorok
    for j in range(3):  # Oszlopok
        cell = cropped_image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
        cells.append(cell)

dominant_colors = [dominant_color(cell) for cell in cells]

for index, color in enumerate(dominant_colors):
    print(f"Négyzet {index + 1}: Domináns szín (BGR) = {color}")

# Új kép létrehozása a domináns színek megjelenítéséhez
result_image = np.zeros((height, width, 3), dtype=np.uint8)
for i in range(3):
    for j in range(3):
        result_image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = dominant_colors[i*3 + j]

cv2.imshow('Domináns Színek', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()