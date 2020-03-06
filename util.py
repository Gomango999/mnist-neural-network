import numpy as np
import math
import _pickle as cPickle

def read_MNIST_label_file(filepath):
	with open(filepath, 'rb') as stream:
		if stream.read(4) != b'\x00\x00\x08\x01':
			raise Exception("Wrong magic number")

		n_items = int.from_bytes(stream.read(4), 'big')

		labels = []
		for i in range(n_items):
			labels.append(np.zeros((10,1)))
			number = int.from_bytes(stream.read(1), 'big', signed=False)
			labels[i][number,0] = 1.0

		return labels

def read_MNIST_image_file_ex(filepath):
	with open(filepath, 'rb') as stream:
		if stream.read(4) != b'\x00\x00\x08\x03':
			raise Exception("Wrong magic number")
		
		n_images = int.from_bytes(stream.read(4), 'big')
		width = int.from_bytes(stream.read(4), 'big')
		height = int.from_bytes(stream.read(4), 'big')

		images = []
		for _ in range(n_images):
			image = np.frombuffer(stream.read(width*height), dtype=np.uint8)
			image = image.reshape((width*height, 1))
			image = image.astype(dtype=np.float) / 255.0
			images.append(image)
			
		return images

def chunk(list, N):
	chunks = []
	num_chunks = math.ceil(len(list)/N)
	for i in range(num_chunks):
		chunks.append(list[i*N:min(i*N+N,len(list))])
	return chunks

def classify(array):
	maxPos = 0
	maxNum = array[0]
	for i in range(1,len(array)):
		if array[i] > maxNum:
			maxNum = array[i]
			maxPos = i
	return maxPos

def save(obj, filename):
    file = open(filename, 'wb')
    cPickle.dump(obj, file, 2)
    file.close()

def load(filename):
    file = open(filename, 'rb')
    obj = cPickle.load(file)
    file.close()          
    return obj




