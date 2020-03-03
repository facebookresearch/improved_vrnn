#
# Copyright (c) Facebook, Inc. and its affiliates.
#
"""
    utils.py
    ~~~~~~~~

    Utilities used across scripts.
"""
# Imports ---------------------------------------------------------------------
import os
import numpy as np
import imageio
import torch
from subprocess import call
import cv2

from PIL import Image, ImageDraw
# -----------------------------------------------------------------------------


def torch2img(t_tensor):

    aux = t_tensor.cpu().data.numpy()
    aux = aux*255
    aux = aux.clip(0, 255)
    aux = aux.astype(np.uint8)

    aux = np.moveaxis(aux, 1, -2)

    aux = np.moveaxis(aux, 0, 1)
    aux = aux.reshape((aux.shape[0], aux.shape[1]*aux.shape[2], -1))

    if aux.shape[0] == 1:
        aux = np.squeeze(aux, 0)
    else:
        aux = aux.transpose((1, 2, 0))

    return aux


def tensor2img(tensor):
    """
    Convert a CxHxW pytorch tensor into a numpy image HxW(xC).
    """
    aux = tensor.cpu().data.numpy()
    aux = aux*255
    aux = aux.clip(0, 255)
    aux = aux.astype(np.uint8)
    aux = np.moveaxis(aux, 0, -1)

    if aux.shape[-1] == 1:
        aux = np.squeeze(aux, -1)

    return aux


def cat_horizontal(img_list):
    """
    Add together images on img_list on a horizontal line.
    """
    if len(img_list[0].shape) == 2:
        return np.concatenate(img_list, 0)
    else:
        return np.concatenate(img_list, 1)


def split_batch_vertically(x):
    """
    Convert a tensor BxCxHxW into a CxB*HxW tensor.
    """
    aux = torch.transpose(x, 0, 1).contiguous()
    aux = aux.view(x.size(1), -1, x.size(3))
    return aux


def split_batch_horizontally(x):
    """
    Convert a tensor BxCxHxW into a CxHxB*W tensor.
    """
    aux = torch.unbind(x, 0)
    aux = torch.cat(aux, 2)
    return aux
    

def save_gif(filename, x, duration=0.25):
    """
    Make a tensor BxTxCxHxW into a GIF.
    """
    images = []
    for timestep in range(x.size(1)):
        img = x[:, timestep]
        img = tensor2img(split_batch_vertically(img))
        images.append(img)

    imageio.mimsave(filename, images, duration=duration, format='gif')
    # np2movie(np.array(images), filename)


def save_gif_np(filename, x, text=None, color_change=10, duration=0.25):
    """
    Make a numpy array BxTxCxHxW into a GIF.
    """
    aux = []

    if text is not None:
        for t in range(len(x)):

            if text is not None:
                if t < 10:
                    color = 'green'
                else:
                    color = 'red'
                cur_text = text[t]
                cur_frame = x[t]
                aux.append(draw_text(add_border(cur_frame, color), cur_text))

    else:
        aux = x

    imageio.mimsave(filename, aux, duration=duration, format='gif')


def draw_text(tensor, text):
    pil = Image.fromarray(tensor)
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return img


def add_border(x, color='green', pad=0):
    w = x.shape[1]
    nc = x.shape[2]
    px = np.zeros((w+2*pad+50, w+2*pad, 3)).astype(np.uint8)
    if color == 'red':
        px[:, :, 0] = 200
    elif color == 'green':
        px[:, :, 1] = 200
    if nc == 1:
        for c in range(3):
            px[pad:w+pad, pad:w+pad, c] = x[:, :, 0]
    else:
        px[pad:w+pad, pad:w+pad, :] = x

    return px

def np2movie(movie_npz, out_path):
    if movie_npz.shape[-1] == 1:
        new = np.ones(movie_npz.shape[:-1]+(3,))
        new[:,:,:,0] = movie_npz[:,:,:,0]
        new[:,:,:,1] = movie_npz[:,:,:,0]
        new[:,:,:,2] = movie_npz[:,:,:,0]
        movie_npz = new
    elif movie_npz.shape[-1] != 3:
        raise ValueError("Shape of channels should be 1 or 3")
        
    # Prepare temp and output dir
    local_dir = "/tmp/ffmpegdir/"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    k = 0
    for i in range(movie_npz.shape[0]):
        im = Image.fromarray(movie_npz[i,:,:,:])
        im.save(local_dir+'temp_'+str(i+1)+'.png')
        k += 1

    call(['ffmpeg', '-y', '-framerate', '4', '-i', local_dir+'temp_%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path])
    call(['rm', '-r', local_dir])
