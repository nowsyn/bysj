import sys
folder, dst = sys.argv[1], sys.argv[2]
root = "/home/nowsyn/bysj/results/"
# all_matched_faces_file = root + "all_matched_faces_{}_in_{}.txt".format(dst, folder)
# all_matched_faces = open(all_matched_faces_file, 'r').readlines()
# all_matched_faces = [int(line.strip('\n')) for line in all_matched_faces]
# total_matched_faces = []
# step  = 5
matched_faces_file = root + "matched_faces_{}_in_{}.txt".format(dst, folder)
report_log = root + "report_{}_in_{}.log".format(dst, folder)
detection_log = root + "detection_in_{}.log".format(folder)
recognition_log = root + "recognition_{}_in_{}.log".format(dst, folder)
matched_faces = open(matched_faces_file, 'r').readlines()
detection_time = open(detection_log, 'r').read()
recognition_time = open(recognition_log, 'r').read()
width, height = 800, 600
with open(report_log, 'w') as f:
	for i in range(0, len(matched_faces)):
		values1 = matched_faces[i].strip('\n').split(',')
		index1 = int(values1[0])
		# values2 = matched_faces[i+1].strip('\n').split(',')
		# index2 =  int(values2[0])
		# if (index2-index1<step): 
		#	for j in range(index1, index2):
		# 		total_matched_faces.append(j)
		x1,y1,x2,y2 = float(values1[1]), float(values1[2]), float(values1[3]), float(values1[4])
		x1,y1,x2,y2 = x1/width,(height-y2)/height,x2/width,(height-y1)/height
		f.write("%s,%f,%f,%f,%f\n"%(index1,x1,y1,x2,y2))
	f.write("detection time: %s s\n" % detection_time)
	f.write("recognition time: %s s\n" % recognition_time)
	# f.write("all matched faces: %d\n" % len(all_matched_faces))
	# f.write("matched faces: %d\n" % len(total_matched_faces))
	# correct = set(all_matched_faces) & set(total_matched_faces)
	# f.write("correct matched faces: %d\n" % len(correct))
	# f.write("accuracy: %f\n" % (len(correct)/float(len(all_matched_faces))))
