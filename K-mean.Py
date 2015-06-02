#*************************************************************************
# clustering using pylab Kmean clustering
# by Maryam , Azima , Mohan
# HW 3 , Data Analytics on Open Cloud EE 5243-004
# Fall 2013
#*************************************************************************

from pylab import imread,imshow,figure,show,subplot
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq
img = imread('frame0000.png')
# reshaping the pixels matrix
pixel = reshape(img,(img.shape[0]*img.shape[1],3))
# performing the clustering
centroids,_ = kmeans(pixel,6) # six colors will be found
# quantization
qnt,_ = vq(pixel,centroids)
# reshaping the result of the quantization
centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
clustered = centroids[centers_idx]
figure(1)
subplot(211)
imshow(flipud(img))
subplot(212)
9
imshow(flipud(clustered))
show()
# visualizing the centroids into the RGB space
from mpl_toolkits.mplot3d import Axes3D
fig = figure(2)
ax = fig.gca(projection='3d')
ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c=centroids/255.,s=100)
show()
