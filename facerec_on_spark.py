from pyspark import SparkContext
import cv2
import numpy as np
import dlib
import time


class Constants:
    FACEREC_HOST = 'http://127.0.0.1:50050/get_descriptors'
    PROJECT_HOME = '/home/nowsyn/bysj'
    VIDEO_FOLDER = '/home/nowsyn/videos'
    VIDEO_SEGMENTS_CSV = '/Input/segments_list.csv'
    SHAPE_PREDICTOR_MODEL = '/home/nowsyn/bysj/models/shape_predictor_68_face_landmarks.dat'
    FACE_RECOGNIZOR_MODEL = '/home/nowsyn/bysj/models/dlib_face_recognition_resnet_model_v1.dat'
    DIMENSION = 96


def preprocess(bgr_img, ratio=0.5):
    import cv2
    shape = bgr_img.shape
    w, h = int(shape[1] * ratio), int(shape[0] * ratio)
    return cv2.resize(bgr_img, dsize=(w, h))


def get_descriptor(img):
    import dlib
    import numpy as np
    shape_model = Constants.SHAPE_PREDICTOR_MODEL
    rec_model = Constants.FACE_RECOGNIZOR_MODEL
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_model)
    recognizor = dlib.face_recognition_model_v1(rec_model)
    bbs, _, _ = detector.run(img, 1, 0.5)
    if len(bbs) != 1:
        raise ValueError("Expect 1 face!")
    shape = predictor(img, bbs[0])
    descriptor = recognizor.compute_face_descriptor(img, shape)
    vec = np.array(map(lambda x: x, descriptor))
    return vec


def get_descriptors_by_rest(img):
    import cPickle
    import requests
    files = {'image': cPickle.dumps(img)}
    res = requests.post(Constants.FACEREC_HOST, files=files)
    return cPickle.loads(str(res.text))


def get_segment_from_line(line):
    s = line.split(',')
    segment = dict()
    prefix = "/home/nowsyn/videos/"
    segment['filename'] = prefix + s[0]
    segment['start'] = float(s[1])
    segment['end'] = float(s[2])
    return segment


def get_faces_from_segment(segment):
    import skvideo.io
    import dlib
    import numpy as np
    shape_model = Constants.SHAPE_PREDICTOR_MODEL
    rec_model = Constants.FACE_RECOGNIZOR_MODEL
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_model)
    recognizor = dlib.face_recognition_model_v1(rec_model)
    video = skvideo.io.LibAVReader(segment['filename'])
    fps = int(round(video.inputfps))
    shape = video.getShape()
    nframes = shape[0]
    k, counter, step = 0, 0, fps
    faces = []
    for frame in video.nextFrame():
        if k == 0:
            img = preprocess(frame, ratio=0.25)
            descriptors = []
            bbs, _, _ = detector.run(img, 1, 0.5)
            for k, bb in enumerate(bbs):
                shape = predictor(img, bb)
                descriptor = recognizor.compute_face_descriptor(img, shape)
                vec = np.array(map(lambda x: x, descriptor))
                face = dict()
                face['segment'] = segment
                face['nframes'] = nframes
                face['pos'] = counter
                face['descriptor'] = vec
                faces.append(face)
        k = (k+1) % step
        counter += 1
    return faces


def get_dst_descriptor():
    import cv2
    dst_img_path = "/home/nowsyn/bysj/images/dst.jpg"
    dst_img = cv2.imread(dst_img_path)
    dst_img = preprocess(dst_img, ratio=0.5)
    dst_descriptor = get_descriptor(dst_img)
    return dst_descriptor


def face_verify(v1, v2, threshold=0.6):
    import numpy as np
    sim = np.linalg.norm(v1-v2)
    return sim<threshold


def get_face_info(face):
    start = face['segment']['start']
    end = face['segment']['end']
    timestamp = start + float(face['pos']) / face['nframes'] * (end-start)
    return timestamp


t1 = time.time()
master = "spark://lenovo-0-0:7077"
sc = SparkContext(master, "Facerec")
lines_rdd = sc.textFile("/Input/segments_list.csv", 32)
segments_rdd = lines_rdd.map(lambda x: get_segment_from_line(x))
faces_rdd = segments_rdd.flatMap(lambda x: get_faces_from_segment(x))
dst_descriptor = get_dst_descriptor()
expected_faces_rdd = faces_rdd.filter(lambda x: face_verify(x['descriptor'], dst_descriptor))
faces_info_rdd = expected_faces_rdd.map(lambda x: get_face_info(x))
timestamps = faces_info_rdd.collect()
t2 = time.time()
print "total execution time: %f s" % (t2-t1)
sc.stop()
with open("/home/nowsyn/bysj/results/timestamps.csv", 'w') as f:
    for ts in timestamps:
        f.write('%f\n'%ts)
