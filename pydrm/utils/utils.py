'''
Utility functions
'''

from time import perf_counter
import numpy as np
# import cv2
# from PIL import Image
import matplotlib.pyplot as plt
# import os
# import re
# import pandas as pd
# import glob

def timeit(method):
    '''
    Timer decorator - measures the running time of a specific function.
    '''
    def timed(*args, **kw):
        print('\n> Starting: {} \n'.format(method.__name__))
        # ts = time.time()
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        # te = time.time()
        print('\n> Timer ({}): {:.2f} sec.'.format(method.__name__, te-ts))
        return result
    return timed

def show(im, cmap=plt.cm.jet, s=12, vmin=None, vmax=None, title=''):
    '''A basic display of the image.'''
    fig, ax = plt.subplots(figsize=(s,s))
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis('off')
    ax.set_title(title)
    plt.show()
    
def splitter(data, sample_size, test_fraction):
    '''
    Splits the data according to a fixed sample size. Separates the sampled
    data into a training and test fraction. Returns the sampled data with the 
    indeces of the training and test samples.
    '''
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data_extract = data[idx[:sample_size]]
    train_size = np.clip(sample_size*(1-test_fraction), 0, sample_size)
    train_size = train_size.astype('int')
    train_sample = data_extract[:train_size]
    test_sample = data_extract[train_size:sample_size]
    idxtr = idx[:train_size]
    idxte = idx[train_size:sample_size]
    return train_sample, idxtr, test_sample, idxte

def shuffler(data, sample_size):
    '''
    Randomly selects a sample of sample_size from the data and returns it 
    with the corresponding indeces.
    '''
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    sample = data[idx[:sample_size]]
    indeces = idx[:sample_size]
    return sample, indeces

def get_xymaps(rx, ry):
    '''
    Produces a flattened mesh grid of X and Y coordinates of resolution (rx,ry)
    '''
    # X-map
    xmap = np.empty((rx*ry))
    for k in range(rx):
        xmap[k*ry:(k+1)*ry] = np.array([k]*ry)
    xmap = xmap.reshape((rx, ry))
    xmap = xmap.ravel()
    # Y-map
    ymap = np.empty((rx*ry))
    for k in range(rx):
        ymap[k*ry:(k+1)*ry] = np.arange(ry)
    ymap = ymap.reshape((rx, ry))
    ymap = ymap.ravel()
    return xmap, ymap

# def resize_stack(data, newx, newy):
#     '''Uses cv2.resize function over a loop'''
#     rx, ry, s0, s1 = data.shape
#     data = data.reshape((rx, ry, s0*s1))
#     data = np.transpose(data, [2,0,1])
#     resized_data = np.empty((data.shape[0], newx, newy), dtype=np.uint8)
#     for k, im in enumerate(data):
#         resized_data[k] = cv2.resize(im, (newy, newx), 
#                                      interpolation=cv2.INTER_CUBIC)
#     resized_data = np.transpose(resized_data, [1,2,0])
#     resized_data = resized_data.reshape((newx, newy, s0, s1))
#     return resized_data

# def fileOpenRB(file, ctype='rgb'):
#     '''Opens file, inverts red and blue channels'''
#     im = np.array(Image.open(file)).astype(np.uint8)
#     if ctype=='rgb':
#         v = im[:,:,2].copy()
#         im[:,:,2] = im[:,:,0]
#         im[:,:,0] = v
#     elif ctype=='grey':
#         im = im[:,:,0]
#     return im

# def glob_open(path, img_type='.jpg'):    
#     '''Glob file opener from iterator and regular expressions'''
#     iterator = sorted(glob.glob(path+f'*{img_type}'), 
#                       key=lambda file:os.path.getctime(file)) 
    
#     fnames, thetas, phis = [], [], []
#     for file in iterator:
#         fname = os.path.basename(file)
#         phi, theta = re.findall('[0-9]+', os.path.splitext(fname)[0])
#         fnames.append(fname)
#         phis.append(phi)
#         thetas.append(theta)
    
#     data_array = pd.DataFrame.from_dict({'fname':fnames, 
#                                          'phi':phis, 
#                                          'theta':thetas})
    
#     data_array.phi = data_array.phi.astype('int')
#     data_array.theta = data_array.theta.astype('int')
    
#     data_array = data_array.sort_values(['phi', 'theta'])
    
#     frames = []
#     for fname in data_array.fname:
#         frames.append(fileOpenRB(path+fname, ctype=None))
#     frames = np.array(frames)
        
#     return frames

# def pngArray(img_file):
#     '''Loads an npy file from an image (.jpg or .png) file.'''
#     return np.array(Image.open(img_file))[...,:3].astype(np.float32)
