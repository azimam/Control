#********************************************************************************
# Using Artificial Neural Networks to predict the direction of the movement of a
# Neutrophil chasing a bacterium in an image
# by Maryam , Azima , Mohan
#*******************************************************************************
from __future__ import division
import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
import scipy.misc
import numpy as np
from sklearn.decomposition import PCA
class1im1=scipy.misc.imread('frame0000.png',flatten=1)
class1im2=scipy.misc.imread('frame0006.png',flatten=1)
class1im3=scipy.misc.imread('frame0013.png',flatten=1)
class1im4=scipy.misc.imread('frame0020.png',flatten=1)
class1im5=scipy.misc.imread('frame0027.png',flatten=1)
class2im1=scipy.misc.imread('frame0216.png',flatten=1)
class2im2=scipy.misc.imread('frame0223.png',flatten=1)
class2im3=scipy.misc.imread('frame0230.png',flatten=1)
class2im4=scipy.misc.imread('frame0237.png',flatten=1)
class2im5=scipy.misc.imread('frame0244.png',flatten=1)
class1vec1=np.reshape(class1im1,np.size(class1im1))
class1vec2=np.reshape(class1im2,np.size(class1im1))
class1vec3=np.reshape(class1im3,np.size(class1im1))
class1vec4=np.reshape(class1im4,np.size(class1im1))
class1vec5=np.reshape(class1im5,np.size(class1im1))
class2vec1=np.reshape(class2im1,np.size(class2im1))
class2vec2=np.reshape(class2im2,np.size(class2im1))
class2vec3=np.reshape(class2im3,np.size(class2im1))
class2vec4=np.reshape(class2im4,np.size(class2im1))
class2vec5=np.reshape(class2im5,np.size(class2im1))
trainData1=np.array([class1vec1,class1vec2,class1vec3,class1vec4,class1vec5,class2vec1,class2vec2,clas
s2vec3,class2vec4,class2vec5])
ncomponents=9
pca = PCA(n_components=ncomponents)
pca.fit(trainData1)
trainData=pca.transform(trainData1)
trainLabels=np.array([1,1,1,1,1,2,2,2,2,2])
trnData = ClassificationDataSet(ncomponents, 1, nb_classes=2)
for i in range(len(trainLabels)):
trnData.addSample(trainData[i,:], trainLabels[i]-1)
tstdata, trndata = trnData.splitWithProportion( 0.40 )
trnData._convertToOneOfMany( )
fnn = buildNetwork( trnData.indim, 20, trnData.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trnData, momentum=0.1, verbose=True, weightdecay=0.01)
for i in range(20):
trainer.trainEpochs(5)
trnresult=percentError(trainer.testOnClassData(),trnData['class'])
print "epoch: %4d" % trainer.totalepochs, \
" train error: %5.2f%%" % trnresult, \
outTrain=fnn.activateOnDataset(trnData)
outTrainLabels=outTrain.argmax(axis=1)+1
numErrTrain=numErr=sum(abs(outTrainLabels!=trainLabels))
accTrain=1-numErrTrain/len(trainLabels)
from __future__ import division
import numpy as np
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
from scipy import diag, arange, meshgrid, where
import scipy.misc
import numpy as np
from sklearn.decomposition import PCA
class1im1=scipy.misc.imread('frame0000.png',flatten=1)
class1im2=scipy.misc.imread('frame0006.png',flatten=1)
class1im3=scipy.misc.imread('frame0013.png',flatten=1)
class1im4=scipy.misc.imread('frame0020.png',flatten=1)
class1im5=scipy.misc.imread('frame0027.png',flatten=1)
class2im1=scipy.misc.imread('frame0216.png',flatten=1)
class2im2=scipy.misc.imread('frame0223.png',flatten=1)
class2im3=scipy.misc.imread('frame0230.png',flatten=1)
class2im4=scipy.misc.imread('frame0237.png',flatten=1)
class2im5=scipy.misc.imread('frame0244.png',flatten=1)
class1vec1=np.reshape(class1im1,np.size(class1im1))
class1vec2=np.reshape(class1im2,np.size(class1im1))
class1vec3=np.reshape(class1im3,np.size(class1im1))
class1vec4=np.reshape(class1im4,np.size(class1im1))
class1vec5=np.reshape(class1im5,np.size(class1im1))
class2vec1=np.reshape(class2im1,np.size(class2im1))
class2vec2=np.reshape(class2im2,np.size(class2im1))
class2vec3=np.reshape(class2im3,np.size(class2im1))
class2vec4=np.reshape(class2im4,np.size(class2im1))
class2vec5=np.reshape(class2im5,np.size(class2im1))
trainData1=np.array([class1vec1,class1vec2,class1vec3,class1vec4,class1vec5,class2vec1,class2vec2,clas
s2vec3,class2vec4,class2vec5])
ncomponents=9
pca = PCA(n_components=ncomponents)
pca.fit(trainData1)
trainData=pca.transform(trainData1)
trainLabels=np.array([1,1,1,1,1,2,2,2,2,2])
trnData = ClassificationDataSet(ncomponents, 1, nb_classes=2)
for i in range(len(trainLabels)):
trnData.addSample(trainData[i,:], trainLabels[i]-1)
tstdata, trndata = trnData.splitWithProportion( 0.40 )
trnData._convertToOneOfMany( )
fnn = buildNetwork( trnData.indim, 20, trnData.outdim, outclass=SoftmaxLayer )
trainer = BackpropTrainer( fnn, dataset=trnData, momentum=0.1, verbose=True, weightdecay=0.01)
for i in range(20):
trainer.trainEpochs(5)
trnresult=percentError(trainer.testOnClassData(),trnData['class'])
print "epoch: %4d" % trainer.totalepochs, \
" train error: %5.2f%%" % trnresult, \
outTrain=fnn.activateOnDataset(trnData)
outTrainLabels=outTrain.argmax(axis=1)+1
numErrTrain=numErr=sum(abs(outTrainLabels!=trainLabels))
accTrain=1-numErrTrain/len(trainLabels)
