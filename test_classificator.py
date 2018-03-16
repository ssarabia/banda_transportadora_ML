from sklearn.ensemble import RandomForestClassifier
from imutils import resize
import pandas as pd
import numpy as np
import time
import cv2


dataX  = pd.read_csv('./dataset/train_x.csv')
y = pd.read_csv('./dataset/train_y.csv')

clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(dataX, np.ravel(y))

candy = ['Jet_azul', 'Flow_negra', 'Flow_blanca', 'Jumbo_naranja', 'Jumbo_roja', 'Chocorramo', 'Fruna_verde', 'Fruna_naranja', 'Fruna_roja', 'Fruna_amarilla']

candy_dict = {'Jet_azul':0, 'Flow_negra':0, 'Flow_blanca':0, 'Jumbo_naranja':0, 'Jumbo_roja':0, 'Chocorramo':0, 'Fruna_verde':0, 'Fruna_naranja':0, 'Fruna_roja':0, 'Fruna_amarilla':0}

def getRGB(image):
    kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    img1 = cv2.imread("dataset/banda.jpeg")
    img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernelOP, iterations=2)
    img2 = image
    img2 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernelCL, iterations=2)
    diff = cv2.absdiff(img2, img1)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    th = 35
    imask =  mask>th
    canvas = np.zeros_like(img2, np.uint8)
    canvas[imask] = img2[imask]
    rprom = 0
    gprom = 0
    bprom = 0
    cont = 0
    a, b, c = canvas.shape
    zero = np.array([0,0,0])
    for i in range(a-1):
        for j in range(b-1):
            arr = canvas[i][j]
            if ((arr > 150).all()):
                bprom += arr[0]
                gprom += arr[1]
                rprom += arr[2]
                cont += 1

    return [int(rprom/cont),int(gprom/cont),int(bprom/cont)]


kernelOP = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
kernelCL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
cap = cv2.VideoCapture(1)
cap.set(3,640)
cap.set(4,480)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE , 0.5)
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
found = False
label = ''
while(True):
    ret, frame = cap.read()
    frame = frame[::, 95:525]
    image = frame
    image = resize(image, width=500)
    image = image[50:3500, 75:480]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    thresh = fgbg.apply(image)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernelOP, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernelCL, iterations=2)
    im, contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0 and cv2.contourArea(contours[0]) > 10000 and cv2.contourArea(contours[0]) < 80000 and found != True:
        found = True
        rect = cv2.minAreaRect(contours[0])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image,[box],0,(0,0,255),2)
        if rect[0][1] > 290 and rect[0][1] < 325:
            area = rect[1][0] * rect[1][1]
            rgb  = getRGB(frame)
            print('Area: ', area)
            print('Color: ', rgb)
            data = rgb + [area]
            label = candy[int(clf.predict([data]))]
            candy_dict[label] += 1
            print(label)
    else:
        found = False
    cv2.putText(image, label, (width-400,height-100) , cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.imshow("objects Found", image)
    time.sleep(0.01)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(candy_dict)
cap.release()
cv2.destroyAllWindows()
