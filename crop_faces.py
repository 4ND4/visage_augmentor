import os
from imutils import face_utils
from mtcnn import MTCNN
import cv2
import dlib

import config

age = 12

class_detector = 'DLIB_CNN'

image_directory = os.path.expanduser('{0}/{1}/'.format(config.IMAGE_DIRECTORY, age))
output_detector = os.path.expanduser('{0}/{1}/'.format(config.OUTPUT_DIRECTORY, class_detector))

if not os.path.exists(output_detector):
    os.mkdir(output_detector)

output_directory = os.path.join(output_detector, str(age))

if not os.path.exists(output_directory):
    os.mkdir(output_directory)


def dlib_detector(img):

    detector = dlib.get_frontal_face_detector()
    rects = detector(img, 1)

    # loop over the face detections
    for (i, rect) in enumerate(rects):

        return face_utils.rect_to_bb(rect)


def dlib_cnn_detector(img):

    WEIGHTS = 'mmod_human_face_detector.dat'

    cnn_face_detector = dlib.cnn_face_detection_model_v1(WEIGHTS)

    dets = cnn_face_detector(img, 1)

    # loop over the face detections

    for i, d in enumerate(dets):
        x = d.rect.left()
        y = d.rect.top()
        w = d.rect.right() - x
        h = d.rect.bottom() - y

        return x, y, w, h


def mtcnn_detector(img):
    detector = MTCNN()

    result_list = detector.detect_faces(img)

    if result_list is not None and len(result_list) != 0:
        # get coordinates

        return result_list[0].get('box')


image_list = os.listdir(image_directory)


dimensions = None

for f in image_list:

    input_path = os.path.join(image_directory, f)

    image = cv2.imread(input_path)

    if class_detector == 'MTCNN':
        dimensions = mtcnn_detector(img=image)

    elif class_detector == 'DLIB':
        dimensions = dlib_detector(img=image)
    elif class_detector == 'DLIB_CNN':
        # image = dlib.load_rgb_image(input_path)
        dimensions = dlib_cnn_detector(img=image)
    else:
        print('detector unavailable')
        exit(0)

    # Needed if you use OpenCV, By default, it use BGR instead RGB

    if dimensions is not None:

        x1, y1, width, height = dimensions

        x2, y2 = x1 + width, y1 + height

        x1 = 0 if x1 < 0 else x1
        x2 = 0 if x2 < 0 else x2
        y1 = 0 if y1 < 0 else y1
        y2 = 0 if y2 < 0 else y2

        cropped_image = image[y1:y2, x1:x2]

        resize_crop = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(output_directory, f), resize_crop)

        print(f, 'cropped')
    else:
        print('face not detected', f)
