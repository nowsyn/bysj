from pyspark import SparkContext
import cv2
import numpy as np
import dlib
import time
import sys


def preprocess(bgr_img, ratio=0.5):
    import cv2
    shape = bgr_img.shape
    w, h = int(shape[1] * ratio), int(shape[0] * ratio)
    return cv2.resize(bgr_img, dsize=(w, h))


def get_descriptor(img):
    import dlib
    import numpy as np
    shape_model = "/home/nowsyn/bysj/models/shape_predictor_68_face_landmarks.dat"
    rec_model = "/home/nowsyn/bysj/models/dlib_face_recognition_resnet_model_v1.dat"
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


def get_face_from_string(data):
    records = data.split(" ")
    label = records[0]
    bb = [int(x) for x in records[1].strip(",").split(",")]
    shape = [int(x) for x in records[2].strip(",").split(",")]
    descriptor = [float(x) for x in records[3].strip(",").split(",")]
    face = dict()
    face["label"] = label
    face["bb"] = bb
    face["shape"] = shape
    face["descriptor"] = descriptor
    return face


def get_dst_descriptor(path):
    import cv2
    dst_img = cv2.imread(path)
    dst_img = preprocess(dst_img, ratio=0.5)
    dst_descriptor = get_descriptor(dst_img)
    return dst_descriptor


def face_verify(v1, v2, threshold=0.5):
    import numpy as np
    sim = np.linalg.norm(v1-v2)
    return sim<threshold


def recognize_on_spark(folder, dst, master="spark://lenovo-0-0:7077"):
    t1 = time.time()
    sc = SparkContext(master, "FaceRecognizor")
    dst_path = "/home/nowsyn/bysj/images/{}.jpg".format(dst)
    dst_descriptor = get_dst_descriptor(dst_path)
    lines_rdd = sc.newAPIHadoopFile(\
        'file:///home/nowsyn/bysj/results/detected_faces_in_{}.txt'.format(folder),\
        'org.apache.hadoop.mapreduce.lib.input.TextInputFormat',\
        'org.apache.hadoop.io.LongWritable',\
        'org.apache.hadoop.io.Text',\
        conf={'textinputformat.record.delimiter': '\n'}\
    )
    faces_rdd = lines_rdd.map(lambda x: get_face_from_string(x[1]))
    matched_faces_rdd = faces_rdd.filter(lambda x: face_verify(x['descriptor'], dst_descriptor))
    matched_faces = matched_faces_rdd.collect()
    out = "/home/nowsyn/bysj/results/matched_faces_{}_in_{}.txt".format(dst, folder)
    with open(out, 'w') as f:
        for face in matched_faces:
            f.write("%s,%s,%s,%s,%s\n"%(face['label'], face['bb'][0], face['bb'][1], face['bb'][2], face['bb'][3]))
    sc.stop()
    t2 = time.time()
    log = "/home/nowsyn/bysj/results/recognition_{}_in_{}.log".format(dst, folder)
    with open(log, 'w') as f:
        f.write("%f"%(t2-t1))


if __name__ == "__main__":
    recognize_on_spark(sys.argv[1], sys.argv[2])
