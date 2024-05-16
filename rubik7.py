import cv2
import numpy as np


def dominant_color(cell):
    # A kép átalakítása 2D-s tömbbé, ahol minden sor egy képpont
    data = np.reshape(cell, (-1, 3))
    
    # Kizárjuk a fekete színt a számításból
    # Fekete szín: [0, 0, 0], az összehasonlítás egy kis küszöbértékkel történik, ha zajos a kép
    non_black = np.all(data > [30, 30, 30], axis=1)  # Ahol az RGB értékek minden csatornán nagyobbak, mint 10
    data = data[non_black]
    
    # Ellenőrizzük, van-e még elegendő adat
    if data.size == 0:
        return None  # Nincs nem fekete szín
    
    # Leggyakoribb szín keresése azok között, amelyek nem feketék
    colors, count = np.unique(data, axis=0, return_counts=True)
    dominant = colors[count.argmax()]
    return dominant

def adjust_gamma(image, gamma=1.5):  # gamma < 1 sötétíti, gamma > 1 világosítja a képet
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def determine_color_bgr(average_bgr, color_bounds_bgr):
    b, g, r = average_bgr
    for color, (lower_bound, upper_bound) in color_bounds_bgr.items():
        if (lower_bound[0] <= b <= upper_bound[0] and
            lower_bound[1] <= g <= upper_bound[1] and
            lower_bound[2] <= r <= upper_bound[2]):
            return color
    return "unknown"  # Ha egyik kategóriába sem esik bele

def match_color(hue, saturation, value, color_bounds):
    for color, bounds in color_bounds.items():
        (lower_hue, lower_sat, lower_val), (upper_hue, upper_sat, upper_val) = bounds
        if lower_hue <= hue <= upper_hue and lower_sat <= saturation <= upper_sat and lower_val <= value <= upper_val:
            return color
    return "unknown"  # If no color matches


i=1
minimum_area = 100
#for i in range (1,6):

imgneve = "minta"+str(i)+".png"
image = cv2.imread(imgneve)

cv2.imshow('eredeti', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#90%os fekete megtartása, minden más fehér - > nem jött be, most 75% -> majdnem jó, 50% .... nehéz olyat találni ami minden mintára megfelelő (itt már inverz)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
#bw_image = cv2.bitwise_not(thresholded_image)
cv2.imshow('Csak 50% fekete', thresholded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#nem akarom rárajzolni a kész képre a kontúrokat
#for contour in contours: 
#    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Megjelenítés
cv2.imshow('Fekete területek kijelölve', image)
cv2.waitKey(0)
cv2.destroyAllWindows()    

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
else:
    largest_contour = None

# Bounding box meghatározása
x, y, w, h = cv2.boundingRect(largest_contour)

# Négyzetes bounding box kiszámítása
side_length = max(w, h)
center_x, center_y = x + w // 2, y + h // 2

# Négyzetes bounding box sarkainak kiszámítása
top_left_x = center_x - side_length // 2
top_left_y = center_y - side_length // 2
bottom_right_x = top_left_x + side_length
bottom_right_y = top_left_y + side_length

# Négyzet kirajzolása
#cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

# Kép megjelenítése
cv2.imshow('Bounding Square', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kivágjuk a bounding box által meghatározott részt
cropped_image = image[y:y+h, x:x+w]

# Eredeti kép, bounding box és kivágott kép megjelenítése
#cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Bounding box kirajzolása az eredeti képre
#cv2.imshow('Original Image with Bounding Box', image)
cv2.imshow('Cropped Image', cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#mivel már nem a 9 négyzet színeinek átlagát veszem, hanem egy megadott területből veszek mintát, fölöslegessé vált a maszkolás, így kivettem ezeket a lépéseket
