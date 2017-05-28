import requests
import cv2
import cPickle
import json
host = "http://127.0.0.1:50050/get_descriptors"
img = "/home/nowsyn/bysj/images/abba.png"
data = cPickle.dumps(cv2.imread(img))
r = requests.post(host, files={'image':data})
print cPickle.loads(str(r.text))
