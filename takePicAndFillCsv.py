import cv2
import HandTrackingModule as htm
import csv

detector = htm.handDetector()
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

filename = 'C:/Users/Administrador/Desktop/neural/Trabalho Bimestral Escobar/data.csv'
csvwriter = csv.writer(open(filename, 'a', newline=''))
img_counter = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    if not success:
        print("failed to grab frame")
        break
    k = cv2.waitKey(1)

    if k != -1:
      print(lmList)
      img_counter = img_counter + 1
      print("Image taken", img_counter)
      formattedLm = []

      for point in lmList:
        for pointIndex in range(4):
          "lmList[0] = [0, 434, 241]"
          newPointX = lmList[pointIndex][1] - point[1] 
          newPointY = lmList[pointIndex][2] - point[2] 
          formattedLm.append(point[1], point[2])          


    cv2.imshow("test", img)
cap.release()
cv2.destroyAllWindows()
