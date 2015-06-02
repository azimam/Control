#*************************************************************************
# clustering using pylab C-mean clustering
# by Maryam , Azima , Mohan
# HW 3 , Data Analytics on Open Cloud EE 5243-004
#*************************************************************************

from numpy import *
from pylab import *
from pylab import plot,show,imread,imshow,figure,show,subplot
from numpy import vstack,array,reshape,uint8,flipud
from numpy.random import random
from numpy.random import rand
from scipy.cluster.vq import vq
import peach as p
# Read the data
img = imread('frame0000.png')
# reshaping the pixels matrix
data = reshape(img,(img.shape[0]*img.shape[1],3))
#print 'This is original data'
#print data
# Number of clusters
K = 4
mu1 = random((76800, 1))
mu2 = random((76800, 1))
10
mu3 = random((76800, 1))
mu = hstack((mu1, 1.-mu1, 1.-mu2, 1.-mu3))
m = 2.00
fcm = p.FuzzyCMeans(data, mu, m)
print "After 20 iterations, the algorithm converged to the centers:"
print fcm(emax=0)
print
print "The membership values are given below:"
print fcm.mu
print
print data
print 'Centroids:'
print fcm.c
# assign each sample to a cluster
qnt,_ = vq(data,fcm.c)
#print
#print idx
# reshaping the result of the quantization
centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
clustered = fcm.c[centers_idx]
figure(1)
subplot(211)
imshow(flipud(img))
subplot(212)
imshow(flipud(clustered))
show()
