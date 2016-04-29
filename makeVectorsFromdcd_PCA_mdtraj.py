#!/usr/bin/env python

"""
author: Guillaume Bouvier
email: guillaume.bouvier@ens-cachan.org
creation date: 01 10 2013
license: GNU GPL
Please feel free to use and modify this, but keep the above information.
Thanks!
"""
        
import IO
import numpy
import itertools
import scipy.spatial
import scipy.stats
import scipy.ndimage.measurements
import SOM
import glob
#from newProtocolModule import *
from SOMTools import *
import cPickle
import os
import ConfigParser
import sys
import PCA
from multiprocessing import Pool
import mdtraj as md
from tqdm import tqdm
from glob import glob

configFileName = sys.argv[1]
Config = ConfigParser.ConfigParser()
Config.read(configFileName)

nframe = Config.getint('makeVectors', 'nframes')
structFile = Config.get('makeVectors', 'structFile')
trajFile = Config.get('makeVectors', 'trajFile')
projection = Config.getboolean('makeVectors', 'projection')
nProcess = Config.getint('makeVectors', 'nProcess')
atoms = Config.get('makeVectors', 'atoms')

pool = Pool(processes=nProcess)

ref = True
descriptorsList = []
eigenVectorsList = []
eigenValuesList = []
meansList = []

top = md.load_topology(structFile)
traj = md.load(glob(trajFile), top = top, atom_indices = top.select(atoms))

print('Loaded topology: \n%s \n' %top)
print('Loaded trajectory: \n%s \n' %traj)

mask = numpy.ones((traj.xyz.shape[1]),dtype="bool")

print('Starting PCA...')
trajIndex = 0
pbar = tqdm(total=nframe, unit='Frame')

while trajIndex < nframe:
  print('Starting %s' %trajIndex)
  distMats = []
  proc = 0
  while proc < nProcess and trajIndex < nframe:
    print('Calculating distance matrix')
    traj_i = traj.xyz[trajIndex,:,:]
    shapeTraj = traj_i.reshape(traj.n_atoms,3)
    print shapeTraj.shape
    dist = scipy.spatial.distance.pdist(shapeTraj)
    distMat = scipy.spatial.distance.squareform(dist)**2
    print('Calculating mean')
    mean = numpy.mean(distMat,axis=1)
    meansList.append(mean)
    distMats.append(distMat)
    proc += 1
    trajIndex += 1
  
  print('Calculating PCA')
  eigenVectors_eigenValues = pool.map(PCA.princomp, distMats)
  print('Calculating eigenVectors')
  eigenVectorsPool = [e[0] for e in eigenVectors_eigenValues]
  print('Calculating eigenValues')
  eigenValuesPool = [e[1] for e in eigenVectors_eigenValues]
  print('Appending eigenValues in lists')
  
  for eigenValues in eigenValuesPool:
    eigenValuesList.append(eigenValues)
  
  print('Appending eigenVectors in lists')
  for eigenVectors in eigenVectorsPool:

    if ref:
      eigenVectors_ref = eigenVectors
      ref = False
      eigenVectors = eigenVectors*numpy.sign(numpy.dot(eigenVectors.T,eigenVectors_ref).diagonal())
      eigenVectorsList.append(eigenVectors.flatten())

    if projection:
      descriptor = numpy.dot(eigenVectors.T,distMat).flatten()

    else:
      descriptor = eigenVectors.T.flatten()
  print('Finished %s' %trajIndex)
  descriptorsList.append(descriptor)
  pbar.update(1)


projections = numpy.asarray(descriptorsList)
numpy.save('projections.npy', projections)
eigenVectorsList = numpy.asarray(eigenVectorsList)
numpy.save('eigenVectorsList', eigenVectorsList)
eigenValuesList = numpy.asarray(eigenValuesList)
numpy.save('eigenValues', eigenValuesList)
meansList = numpy.asarray(meansList)
numpy.save('meansList', meansList)
reconstruction = numpy.concatenate((eigenVectorsList,projections,meansList),axis=1)
numpy.save('reconstruction',reconstruction)

