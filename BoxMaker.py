from imutils.video import VideoStream
import time
import imutils
import cv2
import tensorflow.keras as k
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from numpy import expand_dims
import math
import numpy as np
TIME_PER_CAPTURE = 6
start = (20, 30)  # start point for text used in putText
font = cv2.FONT_HERSHEY_DUPLEX
fontScale = 0.6
color = (0, 0, 0)
thickness = 1
image_size = (224, 224)
classes = 2
shift = 100
kappa, kappa_s = 7, 0
tic = time.time()
vs = VideoStream(src=0).start()
while True:
    toc = time.time()
    frame = vs.read()
    frame = imutils.resize(frame, width=650, height=650)
    frame = cv2.flip(frame, 1)
    time_elapsed = round(toc - tic)
    if time_elapsed == TIME_PER_CAPTURE:
        break
    else:
        cv2.putText(frame, 'Background picture taken in: ' + str(TIME_PER_CAPTURE - time_elapsed), start, font,
                    fontScale, color, thickness)
        cv2.imshow('Take Background Picture', frame)
    cv2.waitKey(1)
cv2.destroyAllWindows()
vs.stop()
background = cv2.resize(frame, image_size)
while True:
    cv2.putText(frame, 'Press q to quit', start, font, fontScale, color, thickness)
    cv2.imshow('Background', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
model = k.models.Sequential([
    k.layers.SeparableConv2D(64, (1, 1), activation='relu', input_shape=(224, 224, 3), depth_multiplier=3),
])
output_layer = 0
outputs = [model.layers[output_layer].output]
box_model = Model(inputs=model.inputs, outputs=outputs)
background_img = img_to_array(background)
background_img = expand_dims(background_img, axis=0)
feature_maps = box_model.predict(background_img)
fmap_back_avg = np.zeros(shape=(feature_maps.shape[1], feature_maps.shape[2]))
span = int(math.sqrt(feature_maps.shape[-1]))
for fmap in feature_maps:
    i = 1
    for _ in range(span):
        for _ in range(span):
            fmap_back_avg += fmap[:, :, i - 1].squeeze()
            i += 1
fmap_back_avg /= (span ** 2)
vs = VideoStream(src=0).start()
sal_flag = False
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=650, height=650)
    frame = cv2.flip(frame, 1)
    input_image = cv2.resize(frame, image_size)
    input_image = img_to_array(input_image)
    input_image = expand_dims(input_image, axis=0)
    feature_maps = box_model.predict(input_image)
    fmap_avg = np.zeros(shape=(feature_maps.shape[1], feature_maps.shape[2]))
    span = int(math.sqrt(feature_maps.shape[-1]))
    for fmap in feature_maps:
        i = 1
        for _ in range(span):
            for _ in range(span):
                fmap_avg += fmap[:, :, i - 1].squeeze()
                i += 1
    fmap_avg /= (span ** 2)
    diff = np.round(fmap_back_avg - fmap_avg, 2)
    sal_diff = np.round(fmap_back_avg - fmap_avg, 2)
    sal_diff[sal_diff <= kappa_s] = 0
    sal_diff[sal_diff > kappa_s] = shift
    diff[diff <= kappa] = 0
    diff[diff > kappa] = shift
    startx, endx, y = [], [], []
    count = 0
    for i in diff:
        if max(i) != 0:
            y.append(count)
            lis = list(i)
            startx.append(lis.index(shift))
            endx.append(len(lis) - list(reversed(lis)).index(shift) - 1)
        count += 1
    startx = np.array(startx)
    startx = (startx / 223 * 650).astype('int')
    endx = np.array(endx)
    endx = (endx / 223 * 650).astype('int')
    y = np.array(y)
    y = (y / 223 * 487).astype('int')
    start, end = (0, 0), (0, 0)
    if not (len(startx) == 0 or len(endx) == 0 or len(y) == 0):
        start = (min(startx), max(min(y), 0))
        end = (max(endx), max(y))
        cv2.rectangle(frame, start, end, color, thickness + 2)
    sal_diff = cv2.resize(sal_diff, (frame.shape[1], frame.shape[0]))
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        break
    elif key == ord('s'):
        sal_flag = not sal_flag
    if sal_flag:
        frame[:, :, 0] = frame[:, :, 0] + sal_diff
    cv2.imshow('Press c to capture image, press s to toggle saliency', frame)
cv2.destroyAllWindows()
vs.stop()
cv2.imwrite('Image.jpg', frame)
f = open('annot.txt', 'w+')
f.write(str(start)+str(end))
f.close()
