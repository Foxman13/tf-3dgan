import sys
import os
import math
import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk
import skimage.io as skio
import scipy.misc as scipymisc

from mpl_toolkits import mplot3d

try:
    import trimesh
    from stl import mesh
except:
    pass
    print ('All dependencies not loaded, some functionality may not work')

LOCAL_PATH = 'c:/Datasets/Shapenet/3DShapeNets/volumetric_data/'
DS_PATH = 'C:/Datasets/ShapeNet/'
SERVER_PATH = '/home/gpu_users/meetshah/3dgan/volumetric_data/'

def getVF(path):
    raw_data = tuple(open(path, 'r'))
    header = raw_data[1].split()
    n_vertices = int(header[0])
    n_faces = int(header[1])
    vertices = np.asarray([map(float,raw_data[i+2].split()) for i in range(n_vertices)])
    faces = np.asarray([map(int,raw_data[i+2+n_vertices].split()) for i in range(n_faces)]) 
    return vertices, faces

def plotFromVF(vertices, faces):
    input_vec = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            input_vec.vectors[i][j] = vertices[f[j],:]
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(input_vec.vectors))
    scale = input_vec.points.flatten(-1)
    axes.auto_scale_xyz(scale, scale, scale)
    plt.show()

def plotFromVoxels(voxels):
    z,x,y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()

def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f, _, _ = sk.marching_cubes(voxels, level=threshold)
    return v, f

def plotMeshFromVoxels(voxels, threshold=0.5):
    v,f = getVFByMarchingCubes(voxels, threshold)
    plotFromVF(v,f)

def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))

def plotFromVertices(vertices):
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)
    axes.scatter(vertices.T[0,:],vertices.T[1,:],vertices.T[2,:])
    plt.show()

def getVolumeFromOFF(path, sideLen=32):
    mesh = trimesh.load(path)
    volume = trimesh.voxel.Voxel(mesh, 0.5).raw
    (x, y, z) = map(float, volume.shape)
    volume = nd.zoom(volume.astype(float), 
                     (sideLen/x, sideLen/y, sideLen/z),
                     order=1, 
                     mode='nearest')
    volume[np.nonzero(volume)] = 1.0
    return volume.astype(np.bool)

def getVoxelFromMat(path, cube_len=64):
    mat = io.loadmat(path)
    voxels = mat['instance']
    voxels = np.pad(voxels,(1,1),'constant',constant_values=(0,0))
    if cube_len != 32 and cube_len == 64:
        voxels = nd.zoom(voxels, (2,2,2), mode='constant', order=0)
    return voxels

def getAll(obj='airplane',train=True, is_local=True, cube_len=64, obj_ratio=1.0):
    objPath = SERVER_PATH + obj + '/30/'
    if is_local:
        objPath = LOCAL_PATH + obj + '/30/'
    objPath += 'train/' if train else 'test/'

    fileList = [f for f in os.listdir(objPath) if f.endswith('.mat')]
    fileList = fileList[0:int(obj_ratio*len(fileList))]
    volumeBatch = np.asarray([getVoxelFromMat(objPath + f, cube_len) for f in fileList],dtype=np.bool)
    return volumeBatch

def getVoxelsFromModelFile(obj='airplane', train=True):
    objPath = DS_PATH + 'train_voxels/' if train else 'val_voxels/'
    objPath = objPath + obj + '/'
    path = objPath + 'model.mat'
    mat = io.loadmat(path)
    voxels = mat['input']
    #voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
    volumeBatch = np.asarray(voxels, dtype=np.bool)
    return volumeBatch

def getImages(obj='000003',train=True):
    images = []

    objPath = DS_PATH + 'train_imgs/' if train else 'val_imgs/' + obj
    objPath = objPath + obj + '/'
    list = os.listdir(objPath)

    for i in range(len(list)):
        img = skio.imread(objPath + list[i])
        images.append(np.array(img))

    return images

def PngToMatrix(pngfilepath, flatten = False):
    """
    """
    imagedata = scipymisc.imread(pngfilepath, False)
    width = len(imagedata[0])
    height = len(imagedata)
    depthinbytes = len(imagedata[0, 0])

    print('Loaded image with size ' + str(width) + 'x' + str(height) + 'x' + str(depthinbytes))

    imageasbitmatrix = [[]]

    if (flatten):
        imageasbitmatrix = np.zeros((width, height))
    else:
        imageasbitmatrix = np.zeros((width, height, depthinbytes * 8))

    depthaverage = 0

    for w in range(0, width):
        for h in range(0, height):
            for d in range(0, depthinbytes):
                value = imagedata[h, w, d]
                #print('[' + str(w) + ', ' + str(h) + ', ' + str(d) + '] == ' + str(value))
                depthaverage += value

                if (d == depthinbytes - 1):
                    if (depthaverage != 0):
                        depthaverage /= depthinbytes # This is essentially making a grayscale translation

                        if (flatten):
                            imageasbitmatrix[w, h] = int(round(depthaverage))
                        else:
                            z = int(math.ceil(depthaverage / 8)) - 1
                            imageasbitmatrix[w, h, z] = 1

                    depthaverage = 0

    return imageasbitmatrix

if __name__ == '__main__':
    path = sys.argv[1]
    volume = getVolumeFromOFF(path)
    plotFromVoxels(volume)
