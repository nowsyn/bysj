import os
import sys
import dlib
import numpy as np
from PIL import Image
import time
import cv2

class FaceRecognizor:
    def __init__(self, face_predictor_model, face_recognizor_model):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_predictor_model)
        self.recognizor = dlib.face_recognition_model_v1(face_recognizor_model)

    def load_img(self, img_path):
        return np.array(Image.open(img_path))

    def preprocess(self, bgr_img, ratio=0.5):
        shape = bgr_img.shape
        w, h = int(shape[1] * ratio), int(shape[0] * ratio)
        return cv2.resize(bgr_img, dsize=(w, h))

    def get_descriptors(self, bgr_img):
        faces = []
        bbs, _, _ = self.detector.run(bgr_img, 0, 0)
        for bb in bbs:
            shape = self.predictor(bgr_img, bb)
            descriptor = self.recognizor.compute_face_descriptor(bgr_img, shape)
            des = np.array([v for v in descriptor])
            faces.append(des)
        return faces

    def compute_similarity(self, vec1, vec2):
        return np.linalg.norm(vec1 - vec2)

    def is_matched(self, v1, v2, t=0.6):
        if self.compute_similarity(v1,v2)<=0.6:
            return True
        else:
            return False


if __name__ == '__main__':
    predictor_path = '../models/shape_predictor_68_face_landmarks.dat'
    face_rec_model_path = '../models/dlib_face_recognition_resnet_model_v1.dat'
    facerec = FaceRecognizor(predictor_path, face_rec_model_path)
    folder, dst = "P1E_S1", "FD"
    labels = []
    t1 = time.time()
    video_path = "/home/nowsyn/videos/{}/all_files.txt".format(folder)
    dst_path = "/home/nowsyn/bysj/images/{}.jpg".format(dst)
    dst_img = cv2.imread(dst_path)
    dst_des = facerec.get_descriptors(dst_img)[0]
    with open(video_path, 'r') as f:
        for line in f.readlines():
            name = line.strip('\n')
            print name
            frame = cv2.imread("/home/nowsyn/videos/{}/{}".format(folder, name))
            faces = facerec.get_descriptors(frame)
            for face in faces:
                if facerec.is_matched(dst_des, face):
                    labels.append(name.strip('.jpg'))
    t2 = time.time()
    print "total time: %f s" % (t2-t1)
    with open("/home/nowsyn/bysj/results/all_matched_faces_{}_in_{}.txt".format(dst, folder), "w") as f:
        for label in labels:
            f.write("%s\n" % label)
