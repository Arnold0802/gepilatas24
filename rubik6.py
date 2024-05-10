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


i=3
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
    #cv2.imshow('Kivágott kép', cropped_image)
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

blurred_image = cv2.bilateralFilter(enhanced_colors, 10, 150, 150)  # Diameter, SigmaColor, SigmaSpace

cv2.imshow('Bilateral Blurred', blurred_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# új megközelítés, a 9 négyzet megkeresése -> nem találom a négyzeteket,próbáltam skippelni a blur-t, a lekerekített sarkok miatt lehet?

gray = cv2.cvtColor(enhanced_colors, cv2.COLOR_BGR2GRAY)

contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#print(len(contours))

#eddig jutottam, feladtam

#9 egyenlő részre osztás tesztelése megint, de most megjelenítem a vonalakat

image = blurred_image # ezzel a szerkesztett képpel folytatom a feldolgozást

# Kép dimenzióinak lekérdezése
height, width = image.shape[:2]

# Vízszintes vonalak
for i in range(1, 3):
    cv2.line(image, (0, i * height // 3), (width, i * height // 3), (0, 255, 0), thickness=2)

# Függőleges vonalak
for i in range(1, 3):
    cv2.line(image, (i * width // 3, 0), (i * width // 3, height), (0, 255, 0), thickness=2)

# Kép megjelenítése
cv2.imshow('Divided Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Kép mentése
#cv2.imwrite('divided_image.jpg', image)


cell_height = height // 3
cell_width = width // 3
margin_height = int(cell_height * 0.33) 
margin_width = int(cell_width * 0.33)

# Minden cella középső részének kivágása és feldolgozása
cells = []
for i in range(3):  # Sorok
    for j in range(3):  # Oszlopok
        cell = blurred_image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
        cells.append(cell)

#for cell in cells:
#    cv2.imshow('cella', cell)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()



for i in range(0,9):


    print(i)
    image = cells[i]
    namestr= "cella"+str(i)
    cv2.imshow(namestr,image)


    height, width = image.shape[:2]

    x = int(width/5)
    y = int(height/5)
    w = int(width/4)
    h = int(height/4)

    cropped_image = image[y:y+h, x:x+w]
    #cv2.imshow('Cropped Image', cropped_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    image = cropped_image
    # Átlagos BGR szín kiszámítása
    average_color_per_row = np.average(image, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    average_color = np.uint8(average_color)  # Konvertálás egész színekre
    print("Átlagos szín (BGR):", average_color)
    print("B: ",average_color[0],"G: ",average_color[1],"R: ",average_color[2])
    b=average_color[0]
    g=average_color[1]
    r=average_color[2]

    # Átlagos szín megjelenítése egy képen
    average_image = np.zeros_like(image, np.uint8)
    average_image[:] = average_color

    #cv2.imshow('Average Color', average_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    # Színek 
    color_bounds_bgr = {
        'white': ((120, 120, 120), (255, 255, 255)),
        'yellow': ((0, 150, 255), (30, 255, 255)),
        'orange': ((0, 100, 200), (50, 180, 255)),
        'blue': ((120, 0, 0), (255, 100, 100)),
        'red': ((0, 0, 120), (75, 75, 255)),
        'green': ((0, 120, 0), (100, 255, 100))
    }

    #color_bounds_hsv = {
    #'white': ((0, 0, 200), (180, 50, 255)),  
    #'yellow': ((0, 100, 100), (45, 255, 200)),  
    #'orange': ((0, 100, 100), (25, 255, 255)),  
    #'blue': ((100, 100, 100), (130, 255, 255)),  
    #'red': ((0, 50, 20), (5, 255, 255)),  
    #'green': ((45, 100, 100), (75, 255, 255))  
    #}



    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #average_hue = np.mean(hsv_image[:,:,0])
    #average_saturation = np.mean(hsv_image[:,:,1])
    #average_value = np.mean(hsv_image[:,:,2])
    #average_hue = int(average_hue)
    #average_saturation = int(average_saturation)
    #average_value = int(average_value)



    #average_color_hsv = match_color(average_hue, average_saturation, average_value, color_bounds_hsv)

    detected_color1 = determine_color_bgr(average_color,color_bounds_bgr)
    max_bgr=max(average_color)
    min_bgr=min(average_color)
    print("max, min:" , max_bgr, min_bgr)
    detected_color="unknown"
    if ((max_bgr - min_bgr) < 30) and (min_bgr>100):
        detected_color = "white"
    if (b<(g//3)) and ((r-g)>30) and (g>70):
        detected_color = "orange"
    if (b<45) and (r>180) and (g>180) and ((g-r<15) and ((r-g<15))):
        detected_color ="yellow"
    if(b>94) and (g<(b//1.8)) and (r<(b//2)) and ((g-r<50) and ((r-g<50))):
        detected_color ="blue"
    if(r>60) and (b<(r//2)) and (g<(r//3)):
        detected_color ="red"
    if (g>30) and (r<(g//2)) and (b<(g//3)):
        detected_color ="green"
    
    
    print("detected color in BGR_bounds is:", detected_color1)
    print("detected color in BGR if-else is:", detected_color)
    #print("HSV values: ",average_hue," ",average_saturation," ",average_value)
    #print(f"The detected color in HSV is: {average_color_hsv}\n")



cv2.waitKey(0)
cv2.destroyAllWindows()