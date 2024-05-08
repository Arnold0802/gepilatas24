import cv2
import numpy as np

def dominant_color(cell):
    # A kép átalakítása 2D-s tömbbé, ahol minden sor egy képpont
    data = np.reshape(cell, (-1, 3))
    
    # Leggyakoribb szín keresése
    colors, count = np.unique(data, axis=0, return_counts=True)
    dominant = colors[count.argmax()]
    return dominant

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

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
else:
    largest_contour = None

# Ha van érvényes kontúr
if largest_contour is not None:
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Kivágás
    cropped_image = image[y:y+h, x:x+w]
    cv2.imshow('Kivágott kép', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#javítani az oldalarányokon?
#(erre nem találtam megoldást)

#9 részre osztás, domináns szín keresés

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

#ne rajzolja meg a kontúrt?
#kontúron kívül eső részek feketék?
