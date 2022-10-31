import cv2
import os
path = 'C:/Users/joaop/Desktop/Atividades/Trabalho Bimestral - Escobar/ValidateImages'

import HandTrackingModule as htm
pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)

img_counter = 0
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True )
    if not success:
        print("failed to grab frame")
        break
    cv2.imshow("Image", img)

    k = cv2.waitKey(1)
    if k != -1:
      formattedLm = []
      print(chr(k))
      path = 'C:/Users/joaop/Desktop/Atividades/Trabalho Bimestral - Escobar/ValidateImages/'
      img_name = chr(k) + "{}.png".format(img_counter)
      cv2.imwrite(os.path.join(path , img_name), img)
      print("{} written!".format(img_name))
      img_counter += 1
    if k%256 == 27:
            print("Escape hit, closing...")
            break

cap.release()
cv2.destroyAllWindows()