import sys
sys.path.insert(0, '/NIRAL/work/ugor/source/brain_classification/Librairies')


import numpy as np

import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import utils 

##############################################################################################Hyperparamters
nb_images = 42 #12,42 #Number of images
flatten = True #True,False #If you want a flatten brain or not for the shape of the 3D object
hemisphere ='left'#'left','right' 

name_gradcam = 'gradcam' #name of the gradcam that you will use to create the 3D brain
name_3Dbrain = '3Dbrain' #name to save the 3D brain 
##############################################################################################

if flatten:
    name_flatten = '_flatten'
else:
    name_flatten = ''
path_brain = '../3DObject/'+hemisphere+'_hemisphere'+name_flatten+'.vtk'

surface = utils.ReadSurf(path_brain)
verts, faces, edges = utils.PolyDataToTensors(surface)

nbr_verts = verts.shape[0]
nbr_faces = faces.shape[0]

Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

name_PF = '../Pix_to_face/PF'+str(nb_images)+'.pt' 
PF = torch.load(name_PF, map_location=torch.device('cpu'))
num_views = PF.shape[0]
image_size = PF.shape[1]
gradcam = torch.load('Saved_gradcam/'+name_gradcam+'.pt', map_location=torch.device('cpu')) 

gradcam_points = torch.zeros(nbr_verts,3)
intermediaire_gradcam_faces = torch.zeros(nbr_faces)
gradcam_faces = torch.zeros(nbr_faces)
gradcam_count = torch.zeros(nbr_faces)

for cam in range(num_views):

    reshape_size = [image_size*image_size]

    PF_image = PF[cam]
    PF_image = PF_image.contiguous().view(reshape_size)

    gradcam_image = gradcam[cam]
    gradcam_image = gradcam_image.contiguous().view(reshape_size)

    intermediaire_gradcam_faces[PF_image] = gradcam_image

    gradcam_faces += intermediaire_gradcam_faces
    gradcam_count[PF_image] += 1.0

    intermediaire_gradcam_faces = torch.zeros(nbr_faces)

zeros_count = ((gradcam_count == 0).nonzero(as_tuple=True)[0])
gradcam_count[zeros_count] = torch.ones(zeros_count.shape[0])
gradcam_faces = torch.div(gradcam_faces,gradcam_count)
gradcam_faces[-1] = gradcam_faces[-2]


for i in range(3):
    ID_verts = faces[:,i]
    gradcam_points[:,i][ID_verts.long()] = gradcam_faces
gradcam_points = torch.max(gradcam_points,dim=1)[0].to(torch.double)


Colors = vtk.vtkDoubleArray()
Colors.SetNumberOfComponents(1)
Colors.SetName('Colors')

for c in range(nbr_verts):
    Colors.InsertNextTypedTuple([gradcam_points[c].item()])
surface.GetPointData().SetScalars(Colors)

###Save 3D brain
writer = vtk.vtkPolyDataWriter()
writer.SetInputData(surface)
title = "Saved_3Dbrain/"+name_3Dbrain+'.vtk' 
writer.SetFileName(title)
writer.Update()
writer.Write()
