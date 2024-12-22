import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# We used 'TkAgg' backend to open the image in new window
matplotlib.use('TkAgg')

# Load the images
blue_image = cv2.imread('C:/mms/blue.jpg')
red_image = cv2.imread('C:/mms/red.jpg')
yellow_image = cv2.imread('C:/mms/yellow.png')
green_image = cv2.imread('C:/mms/green.jpg')
orange_image = cv2.imread('C:/mms/orange.jpg')
brown_image = cv2.imread('C:/mms/brown.png')


def classify_color(image, image_name):
    if image is None:
        print(f"{image_name} not found!")
        return 'Image not found'

    # Blur the image to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV
    color_ranges = {
        'Red': [(0, 50, 50), (10, 255, 255)],  # Kırmızı aralığı
        'Red2': [(160, 50, 50), (180, 255, 255)],  # Kırmızı için ikinci aralık
        'Yellow': [(20, 100, 100), (30, 255, 255)],  # Sarı aralığı
        'Green': [(35, 100, 100), (70, 255, 255)],  # Yeşil aralığı
        'Blue': [(90, 50, 50), (130, 255, 255)],  # Mavi aralığı
        'Orange': [(10, 100, 100), (20, 255, 255)],  # Turuncu aralığı
        'Brown': [(10, 100, 50), (20, 255, 150)]  # Kahverengi aralığı (daha düşük parlaklık)
    }

    max_color = None
    max_pixels = 0

    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # Create a mask for the color
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        count = cv2.countNonZero(mask)

        if count > max_pixels:
            max_pixels = count
            max_color = color

    # Eğer Red ve Red2 birleşirse sadece Red döndür.
    if max_color == 'Red2':
        max_color = 'Red'

    return max_color if max_color else 'Unknown'


# Classify each image
blue_color = classify_color(blue_image, 'Blue M&M')
red_color = classify_color(red_image, 'Red M&M')
yellow_color = classify_color(yellow_image, 'Yellow M&M')
green_color = classify_color(green_image, 'Green M&M')
orange_color = classify_color(orange_image, 'Orange M&M')
brown_color = classify_color(brown_image, 'Brown M&M')  # Kahverengi M&M

# Display the images with color labels
images = [blue_image, red_image, yellow_image, green_image, orange_image, brown_image]
colors = [blue_color, red_color, yellow_color, green_color, orange_color, brown_color]
titles = ['Blue M&M', 'Red M&M', 'Yellow M&M', 'Green M&M', 'Orange M&M', 'Brown M&M']

plt.figure(figsize=(18, 5))
for i in range(6):
    plt.subplot(1, 6, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(f'{titles[i]} - {colors[i]}')
    plt.axis('off')

plt.show()
