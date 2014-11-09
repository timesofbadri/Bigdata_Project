#!/usr/bin/env python
## -*- coding: utf-8 -*-

import sys
import numpy
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
	dict = unpickle('data_batch_1')
	datadict = dict['data']
	size = 100#len(dict['data'])
	featureSize = len(dict['data'][0])/3
	grayMatrix = numpy.zeros((size, featureSize))
	normalRGB = numpy.zeros((size, featureSize,3))

	#Compute intensity and normalize
	items = list(dict.items())
	dataItem=items[0]
	dataMatrix=numpy.array(dataItem[1])
	R=dataMatrix[:,:1024]
	G=dataMatrix[:,1024:2048]
	B=dataMatrix[:,2048:]
	#Construct original picture
	normalRGB = numpy.rollaxis(numpy.asarray([R,G,B]), 0,3)
	#Construct intensity Array
	grayMatrix = (R*0.2989+G*0.5870+B*0.1140)/255

	computePCA(grayMatrix)
    
	#Compute covariance matrix
	#M = numpy.dot(grayMatrix,grayMatrix.T)
	#e,EV = numpy.linalg.eigh(M)
	#normal = normalRGB[0].reshape((32,32,3))
	#plt.imshow(normal) #load
	#plt.show()  # show the window
	#grey = grayMatrix[0].reshape((32,32))
	#plot(grey)

def plot(data):	
	plt.imshow(data, cmap = cm.Greys_r)
	plt.show()

def computePCA(data):
	#todo
			
#Main entry
if __name__ == "__main__":
    main(sys.argv)