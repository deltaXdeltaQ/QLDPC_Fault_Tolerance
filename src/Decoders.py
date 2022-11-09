import numpy as np
import matplotlib.pyplot as plt
from graph_tools import Graph
import networkx as nx
import random
import copy
import time
import json

import ldpc
import bposd

from bposd.css_decode_sim import css_decode_sim
from bposd.hgp import hgp
import pickle

import multiprocessing as mp
import random

from scipy.optimize import curve_fit


# BPOSD decoder
from bposd import bposd_decoder

class BPOSD_Decoder():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float, osd_method:str, osd_order:int):
        self.decoder = bposd_decoder(
                h,
                channel_probs=channel_probs,
                max_iter=max_iter,
                bp_method=bp_method,
                ms_scaling_factor=ms_scaling_factor,
                osd_method=osd_method,
                osd_order=osd_order, )
        self.h = h
    
    def decode(self, synd:np.ndarray):
        self.decoder.decode(synd)
        return self.decoder.osdw_decoding




# First-min BP decoders
from ldpc import bp_decoder

class FirstMinBPDecoder():
    def __init__(self, h:np.ndarray, channel_probs:np.ndarray, max_iter:int, bp_method:str, 
                ms_scaling_factor:float):
        self.decoder = bp_decoder(parity_check_matrix=h,
                                 channel_probs=channel_probs,
                                  max_iter=1,
                                  bp_method=bp_method,
                                  ms_scaling_factor=ms_scaling_factor,)
        self.h = h
        self.max_iter = max_iter
    
    def decode(self, synd:np.ndarray):
        correction = np.zeros(np.shape(self.h)[1])
        current_synd = synd
        iter_counter = 0
        new_correction = self.decoder.decode(current_synd)
        new_synd = (self.h@(new_correction) % 2 + current_synd)%2
        while (np.sum(new_synd) <= np.sum(current_synd)) and (iter_counter < self.max_iter):
            current_synd = new_synd
            correction = (correction + new_correction) % 2
            iter_counter += 1
            
            new_correction = self.decoder.decode(current_synd)
            new_synd = (self.h@(new_correction) % 2 + current_synd)%2
            
        return correction