import os
from Utils.utils import custom_plot
from Utils.StegoSet import StegoSet
from os import listdir
from os.path import isfile, join
from Algorithms.PM1 import insert_message_PM1, extract_message_PM1
from Algorithms.StegoAlgorithm import StegoAlgorithm as algo
from Utils.Experimenter import Experimenter
import matplotlib.pyplot as plt


mypath = os.getcwd() + "/images/grey/"
onlyfiles = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
onlyfiles.remove(mypath + '.DS_Store')

my_set = StegoSet(onlyfiles[:1], init_way='files')

PM1 = algo(insert_message_PM1, extract_message_PM1, {'route'})
exp = Experimenter(my_set, PM1)
exp.make_experiment()

x_vals = exp.percentages

data = {'PSNR': exp.psnr_vals, 'SSIM': exp.ssim_vals, 'BER': exp.ber_vals, 'TIME': exp.time_vals}

for d in data:
    plt.figure(figsize=(10, 5))
    custom_plot(x_vals, data[d])
    
    plt.xlabel('Container fill rate, 100% = 154k bits')
    plt.ylabel('Parameter value')
    plt.xticks(x_vals, rotation = 70)
    plt.title(d)
    
    plt.show()
