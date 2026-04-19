import cv2
from augmentations import adjust_brightness_contrast
from augmentations import frequency_mask
from augmentations import gaussian_blur
from augmentations import time_shift
from augmentations import cutout

img = cv2.imread("test.png")
#adjust_brightness_contrast
if img is None:
    print("Image not found!")
else:
    output = adjust_brightness_contrast(img)

    cv2.imshow("Original", img)
    cv2.imshow("Adjusted", output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    



#frequency_mask
output = frequency_mask(img)

cv2.imshow("Original", img)
cv2.imshow("Masked", output)

cv2.waitKey(0)
cv2.destroyAllWindows()


#gaussian_blur
output = gaussian_blur(img)

cv2.imshow("Blurred", output)

#time_shift

output = time_shift(img)

cv2.imshow("Shifted", output)

# cutout

output = cutout(img)

cv2.imshow("Cutout", output)