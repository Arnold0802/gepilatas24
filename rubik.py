import cv2
import numpy as np


image = cv2.imread('minta1.png')

# HSVre  konvertálás
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Színek 
color_bounds = {
    'white': ((0, 0, 200), (180, 50, 255)),  
    'yellow': ((30, 100, 100), (45, 255, 255)),  
    'orange': ((10, 100, 100), (25, 255, 255)),  
    'blue': ((100, 100, 100), (130, 255, 255)),  
    'red': ((0, 50, 20), (5, 255, 255)),  
    'green': ((45, 100, 100), (75, 255, 255))  
}


results = []

# Színes négyzetek lokalizálása és színazonosítás
for color, (lower, upper) in color_bounds.items():
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # minimum területi szűrő
            x, y, w, h = cv2.boundingRect(contour)
            results.append((color, x, y, w, h))
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Eredmények kiírása és kép megjelenítése
print("Detected colors and their positions:")
for res in results:
    print(res)

cv2.imshow('Detected Squares', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
