from pyspark import SparkContext
import cv2
import numpy as np
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
    shape_model = '/home/nowsyn/bysj/models/shape_predictor_68_face_landmarks.dat'
    rec_model = '/home/nowsyn/bysj/models/dlib_face_recognition_resnet_model_v1.dat'
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


def get_faces_from_frames(data, folder, prefix="/home/nowsyn/videos/"):
    import dlib, cv2
    import numpy as np
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/nowsyn/bysj/models/shape_predictor_68_face_landmarks.dat')
    recognizor = dlib.face_recognition_model_v1('/home/nowsyn/bysj/models/dlib_face_recognition_resnet_model_v1.dat')
    key, frames = data[0], data[1]
    cur, end = 0, len(frames)
    step_s, step_b = 3, 12
    prev, is_prev = -1, True
    faces = []
    while cur < end:
        frame_path = prefix + folder + "/" + frames[cur]
        frame = cv2.imread(frame_path)
        bbs, _, _ = detector.run(frame, 0, 0)
        if len(bbs)>0: 
            if is_prev:
                for bb in bbs:
                    shape = predictor(frame, bb)
                    descriptor = recognizor.compute_face_descriptor(frame, shape)
                    label = frames[cur].strip(".jpg")
                    face = label + " " + "{},{},{},{}".format(bb.left(), bb.top(), bb.right(), bb.bottom()) + " "
                    for p in shape.parts():
                        face += "%d,%d," % (p.x,p.y)
                    face += " "
                    for val in descriptor:
                        face += "%f," % val
                faces.append(face)
                prev = cur
                cur += step_s
            else:
                cur = cur - step_s
            is_prev = True
        else:
            tmp, prev = prev, cur
            if not is_prev:
                cur += step_b
            else:
                cur = tmp + 1
            is_prev = False
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


def detect_on_spark(files_folder, files_list="all_files.txt", master="spark://lenovo-0-0:7077", min_partitions=80, stride=90):
    t1 = time.time()
    sc = SparkContext(master, "FaceDetector")
    in_file = "file:///home/nowsyn/videos/{}/{}".format(files_folder, files_list)
    out_file = "/home/nowsyn/bysj/results/detected_faces_in_{}.txt".format(files_folder)
    lines_rdd = sc.textFile(in_file, min_partitions)
    lines_rdd = lines_rdd.map(lambda x:(int(int(x.strip(".jpg"))/stride), x))
    groups_rdd = lines_rdd.groupByKey().mapValues(list)
    faces_rdd = groups_rdd.flatMap(lambda x: get_faces_from_frames(x, files_folder))
    faces = faces_rdd.collect()
    with open(out_file, "w") as f:
        for face in faces:
             f.write(face+"\n")
    sc.stop()
    t2 = time.time()
    log = "/home/nowsyn/bysj/results/detection_in_{}.log".format(files_folder)
    with open(log, 'w') as f:
        f.write("%f"%(t2-t1))


if __name__ == "__main__":
    detect_on_spark(sys.argv[1])
