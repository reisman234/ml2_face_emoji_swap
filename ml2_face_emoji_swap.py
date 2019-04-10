from os.path import join

import cv2 as cv
import numpy as np
import os
import time 
import argparse

from emotion_classifier import EmotionClassifier

from constants import SIZE_FACE, EMOTIONS


def get_args():
    parser = argparse.ArgumentParser(
        description='This script, mainly aimed as a demonstrator at exhibitions, '
                    'is executed on the Nvidia Jetson TX2 system. '
                    'With this script a task in the field of machine leaning is performed. '
                    'Firstly, an object detector is used to detect faces in a camera stream. '
                    'Secondly, all detected faces are classified by an earlier trained '
                    'artificial neural network in few emotion classes like angry and happy.'
                    'In a last step, the face will now be replaced by a equivalent emoji, '
                    'e.g. if the face is classified as angry the angry emoji will replace the face. '
                    'The resulted image with all replaced faces is than displayed in a window')

    parser.add_argument("--cam-dev",
                        type=int,
                        default=1,
                        required=False,
                        help='The device number of the camera which should be used. By the Default=1 an usb Webcam '
                             'on Jetson TX2 will be used.')

    parser.add_argument("-i", "--image",
                        required=False,
                        help='runs the detection and classification task on an image, mainly as debugging purpose.')

    parser.add_argument("-id", '--image-directory',
                        required=False,
                        help='runs the detection and classification task on a whole directory with multiple images in '
                             'an endless loop. ')

    return parser.parse_args()


class ML2FaceEmojiSwap:

    CASCADE_FILE_DIR = 'haarcascade_files'

    def __init__(self, window_name='ML2-FaceEmotionSwap'):
        self.emoji_images = self.init_emojis()
        self.network = self.init_face_emotion_classifier()
        self.face_cascade, self.cascade_files = self.init_face_cascade()

        self.cam = CameraFrame(window_name=window_name)

    @staticmethod
    def init_emojis():
        emoji_dir = './emojis'
        emoji_images = os.listdir(emoji_dir)
        emoji_images.sort()
        emoji_images = [os.path.join(emoji_dir, f) for f in emoji_images]
        emoji_images = [cv.imread(f, cv.IMREAD_COLOR) for f in emoji_images]
        return emoji_images

    @staticmethod
    def init_face_emotion_classifier():
        network = EmotionClassifier()
        network.load_model()
        return network



    @staticmethod
    def init_face_cascade():

        cascade_files = ['haarcascade_frontalface_alt.xml',
                         'haarcascade_frontalface_alt2.xml',
                         'haarcascade_frontalface_alt_tree.xml',
                         'haarcascade_frontalface_default.xml']
        face_cascade_file = os.path.join(ML2FaceEmojiSwap.CASCADE_FILE_DIR, cascade_files[0])
        return cv.CascadeClassifier(face_cascade_file), cascade_files

    def detect_faces(self, frame):

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # No faces found
        if isinstance(faces,tuple):
            return

        # get the face from the image frame and resize it for prediction
        image_faces = []
        for idx, (x,y,w,h) in enumerate(faces):
            face = gray[y:y+h,x:x+w]
            face = cv.resize(face, (SIZE_FACE, SIZE_FACE), interpolation=cv.INTER_CUBIC) / 255.
            image_faces.append(face)

        faces_for_prediction = np.array(image_faces)
        prediction = self.network.predict(faces_for_prediction)
        prediction = np.round(prediction,3)
        prediction_class = np.argmax(prediction,1)

        # adapted for screen
        detection_result = np.ones((200, 1800), np.uint8)

        # swap each face with its predicted class emoji.
        # create an additional detection result,
        # which shows the cut out face and the model prediction as bar chart.
        for idx, (x, y, w, h) in enumerate(faces):
            emoji = self.emoji_images[prediction_class[idx]]
            emoji = cv.resize(emoji, (w, h))

            roi = frame[y:y+h, x:x+w]

            img2gray = cv.cvtColor(emoji, cv.COLOR_BGR2GRAY)

            ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

            mask_inv = cv.bitwise_not(mask)

            img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

            img2_fg = cv.bitwise_and(emoji, emoji, mask=mask)

            dst = cv.add(img1_bg, img2_fg)
            frame[y:y+h, x:x+w] = dst

            # create for nine faces a detection result
            if idx < 9:
                image_faces[idx] = cv.resize(image_faces[idx], (200, 200)) * 255
                for index, emotion in enumerate(EMOTIONS):

                    cv.putText(image_faces[idx],
                               emotion,
                               (10, index * 20 + 20),
                               cv.FONT_HERSHEY_PLAIN,
                               0.8,
                               (0, 255, 0),
                               1)
                    cv.rectangle(image_faces[idx],
                                 (100, index * 20 + 10),
                                 (100 + int(prediction[idx][index] * 100),
                                 (index + 1) * 20 + 4),
                                 (255, 0, 0),
                                 -1)

                x1 = idx * 200
                y1 = 0

                detection_result[y1:y1+200, x1:x1+200] = image_faces[idx]

        return detection_result

    def load_cascade_file(self, file_idx):
        if not self.cascade_files:
            return
        file_idx -= 49
        cascade_file_name = self.cascade_files[int(file_idx)]
        face_cascade_file = os.path.join(self.CASCADE_FILE_DIR, cascade_file_name)
        print(face_cascade_file)
        self.face_cascade.load(face_cascade_file)

    def run_on_camera(self, cam_dev=0):
        HELP_TEXT = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
        fps = ''
        h_text = HELP_TEXT
        show_help = True
        full_screen = False

        self.cam.open_cam_usb(cam_dev)
        do_exit = False
        while not do_exit:

            start = time.time()

            retval, frame_origin = self.cam.get_frame()
            frame = frame_origin.copy()
            detection_result = self.detect_faces(frame)
            end = time.time()

            if end-start != 0:
                fps = str(round(1 / (end - start), 2))

            key = cv.waitKey(10)
            if show_help:
                h_text = HELP_TEXT + '; FPS: '+fps
            if key == 27:  # ESC key: quit program
                do_exit = True
            elif key == ord('H') or key == ord('h'):  # toggle help message
                show_help= not show_help
            elif key == ord('F') or key == ord('f'):
                full_screen = not full_screen
                self.cam.set_fullscreen(full_screen)

            elif ord('1') <= key <= ord('4'):
                self.load_cascade_file(key)

            self.cam.show_in_window(frame_origin, frame, detection_result,help_text=h_text)

        self.cam.close()

    def run_on_image(self, image_name):

        image_origin = cv.imread(image_name)

        image = image_origin.copy()
        detection_result = self.detect_faces(frame=image)

        self.cam.show_in_window(image_origin,image,detection_result)

        cv.waitKey()
        cv.destroyAllWindows()

    def run_on_image_directory(self, image_directory):
        full_screen = False
        do_exit = False
        help_text = '"Esc" to Quit; "F" to Toggle Fullscreen'
        allowed_image_extension = ['.jpg','.png']
        images = os.listdir(image_directory)
        images = [file for file in images if file.endswith(tuple(allowed_image_extension))]

        image_index = 0
        image_count = len(images)

        while not do_exit:
            image_index = image_index % image_count

            image_origin = cv.imread(join(image_directory,images[image_index]))

            image = image_origin.copy()
            detection_result = self.detect_faces(frame=image)

            self.cam.show_in_window(image_origin, image, detection_result,help_text)

            key = cv.waitKey(5000)
            if key == 27:  # ESC key: quit program
                do_exit = True
            elif key == ord('F') or key == ord('f'):
                full_screen = not full_screen
                self.cam.set_fullscreen(full_screen)

            image_index += 1
            image_index = image_index % image_count

        cv.destroyAllWindows()


class CameraFrame:

    def __init__(self,window_name='Demo_Window', image_width=1920, image_height=1080):
        self.width = image_width
        self.height = image_height
        self.result_frame = np.zeros((1080, 1920, 3), np.uint8)
        self.window_name = window_name
        self.init_window()
        self.cap = None

    def init_window(self):
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.setWindowTitle(self.window_name, self.window_name)
        cv.resizeWindow(self.window_name, self.width, self.height)
        cv.moveWindow(self.window_name, 0, 0)
        cv.imshow(self.window_name,self.result_frame)

    def open_cam_usb(self, dev=0):
        # We want to set width and height here, otherwise we could just do:
        self.cap = cv.VideoCapture(dev)
        return
        # gst_str = ('v4l2src device=/dev/video{} ! '
        #            'video/x-raw, width=(int){}, height=(int){}, '
        #            'format=(string)RGB ! '
        #            'videoconvert ! appsink').format(dev, self.width, self.height)
        # self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    def open_cam_onboard(self):
        # ONLY ON JETSON TX2

        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(self.width, self.height)
        self.cap = cv.VideoCapture(gst_str, cv.CAP_GSTREAMER)

    def get_frame(self):

        retval, frame = self.cap.read()  # grab the next image frame from camera

        return exit, frame

    def set_fullscreen(self,full_screen):
        if full_screen:
            cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
        else:
            cv.setWindowProperty(self.window_name, cv.WND_PROP_FULLSCREEN,cv.WINDOW_NORMAL)

    def show_in_window(self, image_origin, image, detection_result, help_text=None):

        result_frame = np.zeros((1080, 1920, 3), np.uint8)

        image_y, image_x,_ = image_origin.shape

        image_max_x = 1920 // 2
        image_max_y = 800

        image_ratio = image_y / image_x
        new_image_x = image_x
        new_image_y = image_y

        # if image is square or taller than width,
        # resize the image to max height and align width
        if image_y >= image_x:
            # if image_y > image_max_y:
            new_image_y = image_max_y
            new_image_x = int(image_max_y / image_ratio)

        # else: set width to max width and align height
        else:
            new_image_x = image_max_x
            new_image_y = int(image_ratio * image_max_x)

        # adapt the new image sizes to the images
        image_origin = cv.resize(image_origin, (new_image_x, new_image_y))
        image = cv.resize(image, (new_image_x, new_image_y))

        result_frame[0:new_image_y, 0:new_image_x] = image_origin
        result_frame[0:new_image_y, new_image_x:2 * new_image_x] = image

        if detection_result is not None:
            detection_result = cv.cvtColor(detection_result, cv.COLOR_GRAY2RGB)
            result_frame[800:1000, 0:1800] = detection_result
        if help_text:
            cv.putText(result_frame, help_text, (20, 1010), cv.FONT_HERSHEY_PLAIN, 1.0, (32, 32, 32), 4, cv.LINE_AA)
            cv.putText(result_frame, help_text, (20, 1010), cv.FONT_HERSHEY_PLAIN, 1.0, (240, 240, 240), 1, cv.LINE_AA)

        cv.imshow(self.window_name, result_frame)

    def close(self):
        self.cap.release()
        self.cap = None
        cv.destroyAllWindows()


def main():
    args = get_args()

    ml2_fes = ML2FaceEmojiSwap()
    if args.image:
        ml2_fes.run_on_image(args.image)
    elif args.image_directory:
        ml2_fes.run_on_image_directory(args.image_directory)
    else:
        ml2_fes.run_on_camera(args.cam_dev)


if __name__ == '__main__':
    main()

