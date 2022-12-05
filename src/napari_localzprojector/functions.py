import tifffile
import numpy as np
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter, center_of_mass
from napari.types import ImageData

def get_zfilter(im: ImageData, half_size, size_z, size_x, size_y,step_size, method):
    zfilter = np.empty((size_z,int((size_x-half_size)/step_size),int((size_x-half_size)/step_size)))
    i = 0
    if method == 'mean':
        fun = lambda a, b : np.mean(a, b)
    elif method == 'std':
        fun = lambda a, b : np.std(a, b)
    elif method == 'median':
        fun = lambda a, b : np.median(a, b)
    elif method == 'mean_mass': 
        fun = lambda a, b : np.mean(a, b)/center_of_mass(a)[1]
    
    for x in range(0, size_x-half_size, step_size):
        j=0
        for y in range(0, size_y-half_size, step_size):
            temp_im = im[:,x:x+half_size,y:y+half_size]
            zfilter[:,i,j] = fun(temp_im, (1,2))
            j=j+1
        i=i+1
    for i in range(1,size_z):
        zfilter[i-1,:,:] = zfilter[i-1,:,:]-zfilter[i,:,:]
    zfilter[-1,:,:] = zfilter[-1,:,:]/zfilter[-1,:,:]-0.5
    return zfilter
    
def getzmap(zfilter):
    size_z, size_x, size_y = np.shape(zfilter)
    zmap = np.empty((size_x, size_y))
    for i in range(0,size_x):
        for j in range(0,size_y):
            zmap[i,j] = np.where(zfilter[:, i, j] == np.max(zfilter[:,i,j],axis=0))[0][0]
    return zmap

def get_interp_zmap(zmap, size_z, size_x, size_y, step_size, half_size):
    x = np.linspace(0, size_x, int((size_x-half_size)/step_size))
    y = np.linspace(0, size_y, int((size_y-half_size)/step_size))
    x2 = np.linspace(0, size_x, size_x)
    y2 = np.linspace(0, size_y, size_y)
    f = interp2d(x, y, zmap, kind='linear')
    interp_zmap = f(x2,y2)
    return interp_zmap

def get_zproj_loop(im, interp_zmap, size_x, size_y, size_z, dz):
    zproj = np.empty((size_x, size_y))
    for i in range(0, size_x):
        for j in range(0, size_y):
            zind = int(interp_zmap[i, j])
            minz = np.max([zind-dz,0])
            maxz = np.min([zind+dz,size_z])
            zproj[i, j] = np.max(im[minz:maxz, i, j], axis=0)
    return zproj

def get_zproj_linear(im, interp_zmap, size_x, size_y, size_z, dz) -> ImageData:
    row, col = np.indices((size_x,size_y))
    zproj = np.empty((size_z, size_x, size_y))
    zproj_map = np.empty((size_z, size_x, size_y))
    interp_map = np.reshape(np.rint(interp_zmap),(1,size_x*size_y))
    interp_map = interp_map.astype(int)
    row = np.reshape(row,(1,size_x*size_y))
    row = row.astype(int)
    col = np.reshape(col,(1,size_x*size_y))
    col = col.astype(int)
    dz_range = range(-dz,dz+1,1)
    for i in dz_range:
        interp_map[0,np.where(interp_map[0] >= size_z-1)] = size_z-1
        interp_map[0,np.where(interp_map[0] < 0)] = 0
        zproj_map[interp_map, row[0], col[0]] = 1
    zproj = zproj_map*im
    zproj = np.max(zproj,axis=0)
    return zproj