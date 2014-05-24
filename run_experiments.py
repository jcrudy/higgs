'''
Created on May 22, 2014

@author: jason
'''
from models import model_dict
from util import run_experiment
import os


output_dir_base = 'results'
nrows = 5000

output_dir = output_dir_base + '_' + str(nrows)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
run_experiment(model_dict, output_dir, nrows=nrows)
