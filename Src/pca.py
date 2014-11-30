#!/usr/bin/env python
## -*- coding: utf-8 -*-

import sys, getopt
import numpy
import PyML as pyml
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def main(argv):
	#conf = SparkConf().setAppName('pca').setMaster('spark://albhed:7077')
	#sc = SparkContext(conf=conf)
	#inputfile = getArguments(argv)
	#Uncomment following line to enable argument
	#dict = unpickle(inputfile)
	dict = unpickle('data_batch_1')
	dt = unpickle('batches.meta')
	print dt
	print dict['labels'][0]
	#Initialize matrix
	size = len(dict['data'])
	featureSize = len(dict['data'][0])/3
	grayMatrix = numpy.zeros((size, featureSize))
	normalRGB = numpy.zeros((size, featureSize,3))

	#Compute intensity and normalize
	items = list(dict.items())
	dataItem = items[0]
	labels = numpy.array(dict['labels'])
	dataMatrix = numpy.array(dataItem[1])
	R = dataMatrix[:,:1024]
	G = dataMatrix[:,1024:2048]
	B = dataMatrix[:,2048:]
	#Construct original picture
	normalRGB = numpy.rollaxis(numpy.asarray([R,G,B]), 0,3)
	#Construct intensity Array
	grayMatrix = (R*0.2989+G*0.5870+B*0.1140)
	#Normalize
	grayMatrix -= grayMatrix.mean(axis=1)[:, None]

	#Apply PCA
	computePCA(grayMatrix, featureSize, labels)
    
	#Compute covariance matrix
	#M = numpy.dot(grayMatrix,grayMatrix.T)
	#e,EV = numpy.linalg.eigh(M)
	normal = normalRGB[0].reshape((32,32,3))
	plt.imshow(normal) #load
	plt.show()  # show the window
	grey = grayMatrix[0].reshape((32,32))
	plot(grey)

def plot(data):	
	plt.imshow(data, cmap = cm.Greys_r)
	plt.show()

def getArguments(argv):
	if len(sys.argv) != 2:
		print 'Please pass training file'
		sys.exit()
	return str(sys.argv[1])

def computePCA(matrix, featureSize,labels):
	#Compute cov matrix
	cov_mat = numpy.cov(matrix.T)
	print cov_mat.shape
	eig_val_cov, eig_vec_cov = numpy.linalg.eig(cov_mat)# Make a list of (eigenvalue, eigenvector) tuples
	eig_pairs = [(numpy.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

	# Sort the (eigenvalue, eigenvector) tuples from high to low
	eig_pairs.sort()
	eig_pairs.reverse()

	matrix_w = eig_pairs[0][1].reshape(featureSize,1)
	for i in range(200):
		matrix_w = numpy.hstack((matrix_w, eig_pairs[i+1][1].reshape(featureSize,1)))
	print matrix_w.shape

	transformed = matrix.dot(matrix_w)
	print transformed.shape

	data = VectorDataSet(transformed, L = labels)
	print data

	recData = transformed.dot(matrix_w.T) + matrix.mean(axis=1)[:, None]

	plot(recData[0].reshape((32,32)))
			
#Main entry
if __name__ == "__main__":
    main(sys.argv)