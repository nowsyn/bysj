from flask import Flask, request
import numpy as np
import dlib
from constants import Constants
import cPickle
import cv2
app = Flask(__name__)
shape_model = Constants.SHAPE_PREDICTOR_MODEL
rec_model = Constants.FACE_RECOGNIZOR_MODEL
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_model)
recognizor = dlib.face_recognition_model_v1(rec_model)

def preprocess(bgr_img, dim=320):
	shape = bgr_img.shape
	ratio = dim / float(max(shape[1], shape[0]))
	w, h = int(shape[1] * ratio), int(shape[0] * ratio)
	return cv2.resize(bgr_img, dsize=(w, h))

@app.route('/')
def hello_world():
	return True

@app.route('/get_descriptors', methods=['GET', 'POST'])
def get_descriptors():
	descriptors = []
	if request.method == 'POST':
		img = cPickle.loads(request.files['image'].read())
		# img = preprocess(img)
		bbs, scores, idx = detector.run(img, 1, 0.5)
    	for k, bb in enumerate(bbs):
			shape = predictor(img, bb)
			descriptor = recognizor.compute_face_descriptor(img, shape)
			vec = np.array(map(lambda x: x, descriptor))
			descriptors.append(vec)
	return cPickle.dumps(descriptors)

if __name__ == '__main__':
	app.run()
