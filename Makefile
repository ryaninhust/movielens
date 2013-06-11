init:
	pip install https://github.com/douban/dpark/zipball/master
	wget http://www.grouplens.org/system/files/ml-100k.zip
	unzip ml-100k.zip

cluster:
	sudo -u mesos python MovieSimilarities.py -m mesos


