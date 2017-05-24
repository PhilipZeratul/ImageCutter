import cv2
import numpy as np
import sys
import uuid


def ClickAndCrop(event, x, y, flags, param):
    global isCropping
    global startPt, endPt
    global image, imageClone

    if event == cv2.EVENT_LBUTTONDOWN:
        imageClone = image.copy()
        startPt = (x, y)
        isCropping = True
        cv2.imshow("sourceImage", imageClone)

    elif event == cv2.EVENT_MOUSEMOVE:
        if isCropping:
            imageClone = image.copy()
            endPt = (x, y)
            cv2.rectangle(imageClone, startPt, endPt, (0, 255, 0), 2)
            cv2.imshow("sourceImage", imageClone)

    elif event == cv2.EVENT_LBUTTONUP:
        imageClone = image.copy()
        endPt = (x, y)
        isCropping = False
        cv2.rectangle(imageClone, startPt, endPt, (0, 255, 0), 2)
        cv2.imshow("sourceImage", imageClone)


def CropImage(sourceImage):
    global x_0, y_0, x_1, y_1

    MARGIN = 10
    SIZE = 224

    x_01 = max(0, x_0 - MARGIN)
    y_01 = max(0, y_0 - MARGIN)
    x_11 = min(sourceImage.shape[1], x_1 + MARGIN + 1)
    y_11 = min(sourceImage.shape[0], y_1 + MARGIN + 1)

    expandedImage = sourceImage[y_01:y_11, x_01:x_11]
    expandeMargin = abs(expandedImage.shape[0] - expandedImage.shape[1]) / 2

    if expandedImage.shape[0] > expandedImage.shape[1]:  # rows > cols
        expandedImage = \
            cv2.copyMakeBorder(expandedImage, 0, 0, expandeMargin, expandeMargin, cv2.BORDER_REPLICATE)
    else:  # cols > rows
        expandedImage = \
            cv2.copyMakeBorder(expandedImage, expandeMargin, expandeMargin, 0, 0, cv2.BORDER_REPLICATE)

    expandedImage = cv2.resize(expandedImage, (SIZE, SIZE), interpolation=cv2.INTER_AREA)

    cv2.namedWindow("expandedImage")
    cv2.imshow("expandedImage", expandedImage)

    cv2.imwrite("Cropped/" + str(uuid.uuid4()) + ".png", expandedImage)


def FindBoundingRect(mask):
    nonZero = np.nonzero(mask)

    x_0 = np.amin(nonZero[1])
    y_0 = np.amin(nonZero[0])
    x_1 = np.amax(nonZero[1])
    y_1 = np.amax(nonZero[0])

    return x_0, y_0, x_1, y_1


def TrackbarCallBack(callBack):
    global h_min, h_max, s_min, s_max, v_min, v_max
    global hImage, sImage, vImage
    global imageCropped
    global hsvThresImage
    global x_0, y_0, x_1, y_1

    h_min = cv2.getTrackbarPos("H_min", "H")
    h_max = cv2.getTrackbarPos("H_max", "H")
    s_min = cv2.getTrackbarPos("S_min", "S")
    s_max = cv2.getTrackbarPos("S_max", "S")
    v_min = cv2.getTrackbarPos("V_min", "V")
    v_max = cv2.getTrackbarPos("V_max", "V")

    hClone = hImage.copy()
    sClone = sImage.copy()
    vClone = vImage.copy()
    hClone = cv2.inRange(hClone, h_min, h_max)
    sClone = cv2.inRange(sClone, s_min, s_max)
    vClone = cv2.inRange(vClone, v_min, v_max)

    cv2.imshow("H", hClone)
    cv2.imshow("S", sClone)
    cv2.imshow("V", vClone)

    mask = cv2.bitwise_and(hClone, sClone)
    mask = cv2.bitwise_and(mask, vClone)
    # Morphology
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    hsvThresImage = cv2.bitwise_and(imageCropped, imageCropped, mask=mask)

    # Bounding box
    x_0, y_0, x_1, y_1 = FindBoundingRect(mask)
    cv2.rectangle(hsvThresImage, (x_0, y_0), (x_1, y_1), (0, 255, 0), 2)

    cv2.imshow("croppedImage", hsvThresImage)


def DealWithCropped(imageCropped):
        global hImage, sImage, vImage
        global hsvThresImage

        cv2.namedWindow("croppedImage")
        cv2.imshow("croppedImage", imageCropped)

        hsvImageCropped = cv2.cvtColor(imageCropped, cv2.COLOR_BGR2HSV)
        hImage, sImage, vImage = cv2.split(hsvImageCropped)

        cv2.namedWindow("H")
        cv2.createTrackbar("H_min", "H", 0, 180, TrackbarCallBack)
        cv2.createTrackbar("H_max", "H", 180, 180, TrackbarCallBack)
        cv2.imshow("H", hImage)
        cv2.namedWindow("S")
        cv2.createTrackbar("S_min", "S", 0, 255, TrackbarCallBack)
        cv2.createTrackbar("S_max", "S", 255, 255, TrackbarCallBack)
        cv2.imshow("S", sImage)
        cv2.namedWindow("V")
        cv2.createTrackbar("V_min", "V", 0, 255, TrackbarCallBack)
        cv2.createTrackbar("V_max", "V", 255, 255, TrackbarCallBack)
        cv2.imshow("V", vImage)

        while True:
            key = cv2.waitKey(0)

            if key == 32:  # Spacebar
                CropImage(imageCropped)
                break

        cv2.destroyWindow("H")
        cv2.destroyWindow("S")
        cv2.destroyWindow("V")


def main():
    global image, imageClone, imageCropped
    global isCropping

    SCALE = 0.3

    image = cv2.imread("Pictures/" + sys.argv[1])
    image = cv2.resize(image, (0, 0), fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    imageClone = image.copy()

    isCropping = False
    cv2.namedWindow("sourceImage")
    cv2.imshow("sourceImage", image)
    cv2.setMouseCallback("sourceImage", ClickAndCrop)

    while True:
        key = cv2.waitKey(0)

        if key == 32:  # Spacebar
            imageCropped = image[startPt[1]:endPt[1], startPt[0]:endPt[0]]
            DealWithCropped(imageCropped)
        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
