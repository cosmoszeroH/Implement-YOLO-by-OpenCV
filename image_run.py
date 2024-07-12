import numpy as np
import cv2 as cv
import wget
import os


CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5


labels = open("./darknet/cfg/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

cfg_file = '.\darknet\cfg\yolov2-tiny.cfg'
weight_path = 'https://github.com/hank-ai/darknet/releases/download/v2.0/yolov2-tiny.weights'
if not os.path.exists('yolov2-tiny.weights'):
    weight_file = wget.download(weight_path)
else:
    weight_file = 'yolov2-tiny.weights'

net = cv.dnn.readNet(config=cfg_file, model=weight_file)

img_file = 'horse.jpg'
img = cv.imread(img_file)
# cv.imshow('window', img)
# cv.waitKey()

h, w = img.shape[:2]

# the input to the network is called blob object, this following function will transform the image to a blob
# scale factor (1/255.0): to scale the pixel values to [0..1]
# size: the input image will be resized to (416, 416)
# the option swapRB is set True since OpenCV uses BGR
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)


# a blob is a 4D numpy array object (images, channel, width, height)
# the image below shows the red channel of the blob (notice the brightness of the red jacket in the background)
r = blob[0, 0, :, :]
# cv.imshow('blob', r)
# cv.waitKey()

net.setInput(blob)
ln = net.getLayerNames()
# get all the output layers
try:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

outputs = net.forward(ln)

font_scale = 1
thickness = 1
boxes, confidences, class_ids = [], [], []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            class_ids.append(classID)

indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[class_ids[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv.imshow('window', img)
cv.waitKey(0)
cv.destroyAllWindows()
