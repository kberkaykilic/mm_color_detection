import cv2
import numpy as np

# Renk aralıklarını HSV formatında tanımlayın
color_ranges = {
    'Red': [(0, 50, 50), (10, 255, 255)],        # Kırmızı aralığı
    'Red2': [(160, 50, 50), (180, 255, 255)],    # Kırmızı için ikinci aralık
    'Yellow': [(20, 100, 100), (30, 255, 255)],  # Sarı aralığı
    'Green': [(35, 100, 100), (70, 255, 255)],   # Yeşil aralığı
    'Blue': [(90, 50, 50), (130, 255, 255)],      # Mavi aralığı
    'Orange': [(10, 100, 100), (20, 255, 255)],   # Turuncu aralığı
    'Brown': [(10, 100, 50), (20, 255, 150)]      # Kahverengi aralığı (daha düşük parlaklık)
}

def classify_color(hsv_image):
    max_color = None  # En fazla piksel sayısına sahip rengin adı
    max_pixels = 0    # En fazla beyaz piksel sayısı

    # Her renk için maske oluştur ve beyaz piksel sayısını hesapla
    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)  # Alt sınır
        upper_bound = np.array(upper, dtype=np.uint8)  # Üst sınır

        # Renk için maske oluştur
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)  # Belirtilen aralıkta kalan pikselleri maskele
        count = cv2.countNonZero(mask)  # Beyaz piksellerin sayısını say

        # Eğer bu renk için piksel sayısı, en fazla piksel sayısından büyükse güncelle
        if count > max_pixels:
            max_pixels = count  # En fazla piksel sayısını güncelle
            max_color = color    # En fazla pikselli rengi güncelle

    # Eğer tespit edilen renk 'Red2' ise, bu durumu düzelt
    if max_color == 'Red2':
        max_color = 'Red'  # İki aralıktan biri 'Red2' ise sadece 'Red' döndür

    # En fazla beyaz piksel sayısına sahip renk varsa döndür, yoksa 'Unknown' döndür
    return max_color if max_color else 'Unknown'

# Kamerayı aç
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Görüntüyü bulanıklaştır ve HSV uzayına dönüştür
    blurred_image = cv2.GaussianBlur(frame, (5, 5), 0)  # 5x5 boyutunda Gaussian bulanıklaştırma
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)  # BGR'den HSV'ye dönüştür

    # M&M şekerinin rengini bul
    color = classify_color(hsv_image)

    # Renk adını ekrana yaz
    cv2.putText(frame, f'Color: {color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Sonucu göster
    cv2.imshow('M&M Color Detector', frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
