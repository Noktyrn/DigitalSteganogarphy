import numpy as np
import cv2
from .utils import concatenate_image, cut_image_into_blocks

class StegoSet(object):
    def __init__(self, values, init_way='files'):
        n = None
        self.data = []

        if init_way == 'files':
            for path in values:
                test_image = np.float32(np.asarray(cv2.imread(path, cv2.IMREAD_GRAYSCALE)))
                if not n:
                    n = len(test_image)
                    self._shape = n
                
                if len(test_image[0]) != n or len(test_image) != n:
                    raise ValueError('All images must be of the same size')
                
                self.data.append(test_image)
        elif init_way == 'images':
            self.data = values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        self.__i = 0
        return iter(self.data)
    
    def __next__(self):
        if self.__i<len(self.data)-1:
            self.__i += 1         
            return self.data[self.__i]
        else:
            raise StopIteration
    
    def cut(self, n=8):
        if len(self.data[0].shape) == 2:
            for i in range(len(self.data)):
                self.data[i] = cut_image_into_blocks(self.data[i], n)
    
    def concat(self):
        if len(self.data[0].shape) == 4:
            for i in range(len(self.data)):
                self.data[i] = concatenate_image(self.data[i])
