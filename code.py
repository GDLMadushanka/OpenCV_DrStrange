import cv2
import imutils

l_img = cv2.imread("power1.png", cv2.IMREAD_UNCHANGED)
s_img = imutils.resize(l_img, 300)

hand_cascade = cv2.CascadeClassifier('closed_frontal_palm.xml')

cap = cv2.VideoCapture(0)  # creating camera object

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hands = hand_cascade.detectMultiScale(gray)

    for (x,y,w,h) in hands:
        if(w>120 and h>120):

            y1, y2 = y-70, y-70 + s_img.shape[0]
            x1, x2 = x-70, x-70 + s_img.shape[1]

            alpha_s = s_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                frame[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] + alpha_l * frame[y1:y2, x1:x2, c])

    cv2.imshow('img',frame)

    k = cv2.waitKey(10) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
