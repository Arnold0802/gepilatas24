import cv2
import numpy as np

#def dominant_color(cell):
#    # A kép átalakítása 2D-s tömbbé, ahol minden sor egy képpont
#    data = np.reshape(cell, (-1, 3))
#    
#    # Leggyakoribb szín keresése
#    colors, count = np.unique(data, axis=0, return_counts=True)
#    dominant = colors[count.argmax()]
#    return dominant

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


i=3
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

#maszk készítése, hogy a kockán kívül eső részek legyenek feketék
height, width = image.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)
cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

cv2.imshow('Maszk', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

#maszk alkalmazása a képre

masked_image = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('Maszkolt kép', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Ha van érvényes kontúr
if largest_contour is not None:
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Kivágás
    cropped_image = masked_image[y:y+h, x:x+w]
    cv2.imshow('Kivágott kép', cropped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
else:
    Exception("nincs kontúr")

#javítani az oldalarányokon?
#(erre nem találtam megoldást)

#új lépés, színek élénkítése (cropped image-el megyek tovább)
#result = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)  # HSV színtérbe konvertálás
#result[:, :, 1] = cv2.add(result[:, :, 1], 50)  # Szín telítettségének növelése
#result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)  # Visszaalakítás BGR-be
#
#cv2.imshow('Enhanced Colors', result)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
#hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])  # V csatorna (fényerő) hisztogram kiegyenlítése
#enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#cv2.imshow('Histogram Equalized', enhanced_img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#gamma_corrected = adjust_gamma(cropped_image, gamma=1.5)
#
#cv2.imshow('Gamma Corrected', gamma_corrected)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#
#lab = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2LAB)  # LAB színtérbe konvertálás
#l, a, b = cv2.split(lab)
#cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Kontraszt korlátozott adaptív hisztogram kiegyenlítő
#l = cl.apply(l)
#updated_lab = cv2.merge((l, a, b))
#enhanced_img = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)
#
#cv2.imshow('Contrast Enhanced', enhanced_img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

#1. lépés: gamma korrekció

gamma_corrected = adjust_gamma(cropped_image, gamma=1.9)

cv2.imshow('Gamma Corrected', gamma_corrected)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#2. lépés 

enhanced_colors = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2HSV)  # HSV színtérbe konvertálás
enhanced_colors[:, :, 1] = cv2.add(enhanced_colors[:, :, 1], 50)  # Szín telítettségének növelése
enhanced_colors = cv2.cvtColor(enhanced_colors, cv2.COLOR_HSV2BGR)  # Visszaalakítás BGR-be

cv2.imshow('Enhanced Colors', enhanced_colors)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

blurred_image = cv2.bilateralFilter(enhanced_colors, 20, 250, 250)  # Diameter, SigmaColor, SigmaSpace

cv2.imshow('Bilateral Blurred', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





