import cv2
import matplotlib.pyplot as plt

path = 'foots.png'
image = cv2.imread(path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.subplot(1, 1, 1)
# plt.imshow(img_gray)
# plt.show()

xml_data = cv2.CascadeClassifier('haarcascade_fullbody.xml')
detecting = xml_data.detectMultiScale(image_gray, 1.1, 4)

for (x, y, w, h) in detecting:
    cv2.rectangle(image_gray, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.imshow('image_gray', image_gray)
cv2.waitKey(0)



# detects humans

# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#
# image = cv2.imread('foots.png')
#
# (humans, _) = hog.detectMultiScale(image,
#                                    winStride=(2, 2),
#                                    padding=(2, 2),
#                                    scale=1.21)
#
# for (x, y, w, h) in humans:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
#
# cv2.imshow('image', image)
# cv2.waitKey(0)
