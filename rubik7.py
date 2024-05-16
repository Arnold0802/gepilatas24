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

def find_closest_color(color, colors):
    min_distance = float('inf')
    closest_color_name = None
    for name, value in colors.items():
        distance = np.linalg.norm(color - value)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
    return closest_color_name

############################################################################################################################################

##########          HLS

# RGB színek HLS színtérbe konvertálása
def rgb_to_hls(rgb_color):
    rgb_color = np.reshape(rgb_color, (1, 1, 3)).astype(np.uint8)
    hls_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HLS)
    return hls_color[0, 0, :]

def find_closest_color_hls(color, colors_hls):
    color_hls = rgb_to_hls(color)
    min_distance = float('inf')
    closest_color_name = None
    
    for name, value in colors_hls.items():
        distance = np.linalg.norm(color_hls - value)
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
            
    return closest_color_name

############################################################################################################################################

i=2
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
#cv2.imshow('Csak 50% fekete', thresholded_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#nem akarom rárajzolni a kész képre a kontúrokat
#for contour in contours: 
#    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

# Megjelenítés
#cv2.imshow('Fekete területek kijelölve', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()    

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
#cv2.imshow('Bounding Square', image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Kivágjuk a bounding box által meghatározott részt
cropped_image = image[y:y+h, x:x+w]

# Eredeti kép, bounding box és kivágott kép megjelenítése
#cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Bounding box kirajzolása az eredeti képre
#cv2.imshow('Original Image with Bounding Box', image)
cv2.imshow('Cropped Image', cropped_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#mivel már nem a 9 négyzet színeinek átlagát veszem, hanem egy megadott területből veszek mintát, fölöslegessé vált a maszkolás, így kivettem ezeket a lépéseket


gamma_corrected = adjust_gamma(cropped_image, gamma=1.9)

cv2.imshow('Gamma Corrected', gamma_corrected)
cv2.waitKey(0)
cv2.destroyAllWindows()
image = gamma_corrected # ezzel a szerkesztett képpel folytatom a feldolgozást

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
        cell = image[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
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
    cv2.imshow(f'Cropped Image{i}', cropped_image)
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

    #cv2.imshow(f'Average Color{i}', average_image)



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


    # Definiáljuk a 6 szín RGB kódját
    colors = {
        'red': np.array([255, 0, 0]),
        'green': np.array([0, 255, 0]),
        'blue': np.array([0, 0, 255]),
        'yellow': np.array([255, 255, 0]),
        'orange': np.array([255, 165, 0]),
        'white': np.array([255, 255, 255])
    }

    # Színek HLS színtérben
    colors_hls = {name: rgb_to_hls(color) for name, color in colors.items()}

    # Például adott szín (RGB)
    test_color = np.array([r, g, b])

    # Legközelebbi szín megtalálása
    closest_color = find_closest_color(test_color, colors)
    print(f"RGB The closest color to {test_color} is {closest_color}")

    closest_color_hls = find_closest_color_hls(test_color, colors_hls)
    print(f"HLS The closest color to {test_color} in HLS space is {closest_color_hls}")





cv2.waitKey(0)
cv2.destroyAllWindows()