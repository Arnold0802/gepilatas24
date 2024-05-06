import cv2
import numpy as np
import matplotlib.pyplot as plt

# Kép betöltése
image = cv2.imread('minta1.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Kép megjelenítése (opcionális)
plt.imshow(image_rgb)
plt.show()

# Színek kinyerése
def extract_colors(image, num_colors):
    # A kép átalakítása adatok listájává
    data = np.reshape(image, (-1, 3))
    
    # K-means klaszterezés a színek csoportosításához
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.6)
    _, labels, centers = cv2.kmeans(data.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # A leggyakoribb színek meghatározása
    _, counts = np.unique(labels, return_counts=True)
    most_frequent = centers[counts.argsort()[::-1]]
    
    return most_frequent.astype(int)

# Színkinyerés futtatása
colors = extract_colors(image_rgb, 5)  # 5 leggyakoribb szín
print("Found colors (RGB):")
print(colors)
