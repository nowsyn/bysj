#!/usr/bin/env sh
folder=$1
dst=$2
detected_faces=/home/nowsyn/bysj/results/detected_faces_in_"$folder".txt
matched_faces=/home/nowsyn/bysj/results/matched_faces_"$dst"_in_"$folder".txt
report_log=/home/nowsyn/bysj/results/report_"$dst"_in_"$folder".log

if [ ! -f "$detected_faces" ]; then
spark-submit --master=yarn --deploy-mode client --num-executors 20 --executor-cores 4 --executor-memory 8G /home/nowsyn/bysj/src/detector_on_spark.py $folder 2>/dev/null
fi

# if [ ! -f "$matched_faces" ]; then
spark-submit --master=yarn --deploy-mode client --num-executors 20 --executor-cores 4 --executor-memory 8G /home/nowsyn/bysj/src/recognizor_on_spark.py $folder $dst 2>/dev/null
# fi

# if [ ! -f "$report_log" ]; then
python /home/nowsyn/bysj/src/report.py $folder $dst
# fi
