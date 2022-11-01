import cv2
import HandTrackingModule as htm
import csv

detector = htm.handDetector()
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)

filename = 'C:/Users/Administrador/Desktop/neural/data.csv'
csvwriter = csv.writer(open(filename, 'a', newline=''))
img_counter = 0


while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=False)

    sizes = detector.getNewSizes(img)

    if not success:
        print("failed to grab frame")
        break
    k = cv2.waitKey(1)

    if k != -1:
        img_counter = img_counter + 1
        formattedLm = []
        for lm in lmList:
            formattedLm.append(lm[1])
            formattedLm.append(lm[2])
    cropped = img[sizes[0]:sizes[1], sizes[3]:sizes[2]]
    if(cropped.shape[0] > 0 and cropped.shape[1] > 0):
        cv2.imshow("test2", cropped)

    cv2.imshow("test", img)

cap.release()
cv2.destroyAllWindows()
