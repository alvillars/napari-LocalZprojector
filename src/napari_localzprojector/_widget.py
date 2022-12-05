"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING
from .functions import get_interp_zmap, get_zfilter, get_zproj_linear, getzmap
from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from napari.types import ImageData
import numpy as np
from napari.utils.notifications import show_info

@magic_factory( call_button="run", 
                halfsize={"widget_type": "SpinBox","value":40, "name":'spin', "label":'sxy:', "max":100}, 
                step_size={"widget_type": "SpinBox","value":20, "name":'spin1', "label":'dxy:', "max":100},
                dz={"widget_type": "SpinBox","value":1, "name":'spin2', "label":'dz:', "max":5},
                minT={"widget_type": "SpinBox","value":0, "name":'spin3', "label":'minT:', "max":10},
                maxT={"widget_type": "SpinBox","value":0, "name":'spin4', "label":'maxT:', "max":10},
                minZ={"widget_type": "SpinBox","value":0, "name":'spin5', "label":'minZ:', "max":60},
                maxZ={"widget_type": "SpinBox","value":0, "name":'spin6', "label":'maxZ:', "max":60},
                dropdown={"choices":  ['mean','std','median','mean_mass','mean_std']})          

def localzprojection(layer : ImageData, halfsize=40, step_size=20, dz=1, minT=1, maxT=1, minZ=0, maxZ=0, dropdown='mean') -> ImageData:
    layer = np.squeeze(layer)

    if np.ndim(layer) < 3:
        show_info('this is not a 3D stack, no need for projection')

    if np.ndim(layer) == 3:
        show_info(str(np.shape(layer)))
        size_z, size_x, size_y = np.shape(layer)
        zfilter = get_zfilter(im=layer[minZ:maxZ,:,:], half_size=halfsize, size_z=size_z,
                    size_x=size_x, size_y=size_y, step_size=step_size, method=dropdown)+minZ
        zmap = getzmap(zfilter)
        interp_zmap = get_interp_zmap(zmap= zmap, size_z=size_z, 
                    size_x=size_x, size_y=size_y, step_size=step_size, half_size=halfsize)
        zproj = get_zproj_linear(im=layer, interp_zmap=interp_zmap, 
                    size_z=size_z, size_x=size_x, size_y=size_y, dz=dz)
        return zproj

    if np.ndim(layer) > 3:
        show_info(str(np.shape(layer)))
        size_t, size_z, size_x, size_y = np.shape(layer)
        zproj = np.empty((size_t, size_x, size_y))
        for t in range(minT, maxT):
            show_info(str(t))
            temp_im = np.reshape(layer[t,minZ:maxZ,:,:],(maxZ-minZ, size_x, size_y))
            zfilter = get_zfilter(im=temp_im, half_size=halfsize, size_z=size_z,
                        size_x=size_x, size_y=size_y, step_size=step_size, method=dropdown)+minZ
            zmap = getzmap(zfilter)
            interp_zmap = get_interp_zmap(zmap= zmap, size_z=size_z, 
                        size_x=size_x, size_y=size_y, step_size=step_size, half_size=halfsize)
            zproj[t,:,:] = get_zproj_linear(im=layer[t,:,:,:], interp_zmap=interp_zmap, 
                        size_z=size_z, size_x=size_x, size_y=size_y, dz=dz)
        zproj = np.reshape(zproj,(size_t, 1, size_x, size_y))
        return zproj