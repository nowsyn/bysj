#!/usr/bin/env sh
slave_file="/home/nowsyn/spark/conf/slaves"
function start_server()
{
	echo $1
	ssh -n $1 "export FLASK_APP=/home/nowsyn/bysj/src/start_facerec.py; python -m flask run --port=50050 > /dev/null 2>&1 &"
}
while read line
do
	if echo $line | grep -qe '^lenovo'
		then start_server $line
	fi
done < $slave_file
