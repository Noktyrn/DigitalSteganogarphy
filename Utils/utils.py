import numpy as np
import cv2
from math import log10, sqrt
import matplotlib.pyplot as plt

qm = np.array( [[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

def count_BER(a, b):
    ber = 0
    for x, y in zip(a, b):
        if x != y:
            ber += 1
    
    return ber / len(a)


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def cut_image_into_blocks(im, n=8):
    for s in im.shape:
        if s % n != 0:
            raise RuntimeError('Shapes do not match: {} % {} != 0'.format(s, n))
    
    rows = im.shape[0] // n
    res = [[] for _ in range(rows)]
    
    for i in range(rows):
        row_ind = i*n
        for j in range(rows):
            col_ind = j*n
            res[i].append(im[row_ind:row_ind+n, col_ind:col_ind+n])
            
    return np.asarray(res)


def concatenate_image(im):
    res = []
    for r in im:
        res.append(np.concatenate(r, axis=1))
        
    return np.asarray(np.concatenate(res, axis=0))


def dct2d(a):
    return cv2.dct(a)


def idct2d(a):
    return np.around(cv2.idct(a))


def get_dct_coefs(im):
    dct_blocks = [[] for row in im]

    for i in range(len(im)):
        for block in im[i]:
            dct_blocks[i].append(dct2d(block))
    
    return np.asarray(dct_blocks)


def get_image_from_dct_coefs(im):
    idct_blocks = [[] for row in im]

    for i in range(len(im)):
        for block in im[i]:
            idct_blocks[i].append(idct2d(block))
    
    return np.asarray(idct_blocks)


def get_dct_coefs(im):
    dct_blocks = [[] for row in im]

    for i in range(len(im)):
        for block in im[i]:
            dct_blocks[i].append(dct2d(block))
    
    return np.asarray(dct_blocks)


def get_image_from_dct_coefs(im):
    idct_blocks = [[] for row in im]

    for i in range(len(im)):
        for block in im[i]:
            idct_blocks[i].append(idct2d(block))
    
    return np.asarray(idct_blocks)

def custom_plot(x, y, ax=None, **plt_kwargs):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plt_kwargs) ## example plot here
    return(ax)


def quantize_dct_blocks(blocks):
    res = [[] for _ in range(len(blocks))]
    for i in range(len(res)):
        for b in blocks[i]:
            res[i].append(b/qm)
    
    return np.asarray(res)

def dequantize_dct_blocks(blocks):
    res = [[] for _ in range(len(blocks))]
    for i in range(len(res)):
        for b in blocks[i]:
            res[i].append(b*qm)
    
    return np.asarray(res)

