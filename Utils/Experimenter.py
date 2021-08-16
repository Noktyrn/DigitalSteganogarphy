import time
import numpy as np
from .utils import PSNR, count_BER
from skimage.metrics import structural_similarity as ssim

class Experimenter(object):
    def __init__(self, dataset, algo, MAX_VALUE=153843, PERCENTAGES=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):
        self.data = dataset
        self.algorithm = algo
        self.max_val = MAX_VALUE
        self.percentages = PERCENTAGES

        self.psnr_vals = []
        self.ssim_vals = []
        self.ber_vals = []
        self.time_vals = []
    
    def make_experiment(self):
        self.psnr_vals = []
        self.ssim_vals = []
        self.ber_vals = []
        self.time_vals = []

        for p in self.percentages:
            m_len = round(p * self.max_val)
            m = np.random.randint(0, 2, size=int(m_len))
            
            print(len(m))
            
            p = []
            s = []
            b = []
            t = []
            for im in self.data:
                start = time.time()
                emb_im, extract_data = self.algorithm.insert_message(im, m)
                new_m = self.algorithm.extract_message(emb_im, extract_data)
                end = time.time()
                
                psnr = PSNR(im, emb_im)
                ssim_noise = ssim(im, emb_im, data_range=emb_im.max() - emb_im.min())
                ber = count_BER(m, new_m)

                t.append(end-start)
                p.append(psnr)
                s.append(ssim_noise)
                b.append(ber)
            
            self.psnr_vals.append(np.mean(p))
            self.ssim_vals.append(np.mean(s))
            self.ber_vals.append(np.mean(b))
            self.time_vals.append(np.mean(t))
