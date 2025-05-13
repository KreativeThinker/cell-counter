import cv2

img = cv2.imread("./data/Ananya Sehgal_T1C1/Test 1/Slide 1/Au1_R2_T1m.jpg")
blue = img[:, :, 0]

# CLAHE to normalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
blue_eq = clahe.apply(blue)

# Blur + adaptive threshold
blur = cv2.GaussianBlur(blue_eq, (3, 3), 0)
thresh = cv2.adaptiveThreshold(
    blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
)

# Invert if needed
thresh = cv2.bitwise_not(thresh)

# Morph cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Contours
cnts = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

out = img.copy()
count = 0
for c in cnts:
    area = cv2.contourArea(c)
    if area > 50:
        cv2.drawContours(out, [c], -1, (0, 255, 0), 1)
        count += 1

print("Cells:", count)

display = cv2.resize(out, (800, 600))
cv2.imshow("Contours", display)
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
