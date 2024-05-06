import cv2
import numpy as np


image = cv2.imread('minta1.png')
original = image.copy()

# Szürkeárnyalatos konvertálás és Gauss-szűrő
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Éldetektálás
edges = cv2.Canny(blurred, 50, 150)

# Kontúrkeresés
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Nagy négyzet keresése és kijelölése
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 1000:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 0.95 < aspect_ratio < 1.05:  # Négyzet keresése
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi = original[y:y+h, x:x+w]  # Kivágjuk a nagy négyzetet

                # Most a kivágott részen belül keressük a kicsi négyzeteket
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blurred_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                edges_roi = cv2.Canny(blurred_roi, 50, 150)
                contours_roi, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours_roi:
                    area = cv2.contourArea(cnt)
                    if area > 50:  # Kis négyzet területe
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                        if len(approx) == 4:
                            x2, y2, w2, h2 = cv2.boundingRect(approx)
                            cv2.rectangle(roi, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

# Megjelenítés
cv2.imshow('Rubik Cube with small squares', roi)
cv2.imshow('Original Image with large square', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
