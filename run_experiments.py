'''
Created on May 22, 2014

@author: jason
'''
from models import model_dict
from util import run_experiment


output_dir = 'results'
nrows = 5000
run_experiment(model_dict, output_dir, nrows=nrows)
