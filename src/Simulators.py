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
from ldpc import bp_decoder
from bposd import bposd_decoder

from scipy.optimize import curve_fit
import stim

import sys
sys.path.append("./")
# from Decoders import BPOSD_Decoder, FirstMinBPDecoder
from Decoders import *

# sys.path.append("/Users/qian/Box Sync/Research/Jiang_group_projects/Squeezed_Cats_Synchronized/Squeezed_Cat/Codes/src/")
sys.path.append("./")
from ErrorPlugin import *

from CircuitScheduling import ColorationCircuit, RandomCircuit

# Modify the multiprocessing functions
def fun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))


def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()

    proc = [mp.Process(target=fun, args=(f, q_in, q_out))
            for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()

    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]

    [p.join() for p in proc]

    return [x for i, x in sorted(res)]


## Save the chosen hgps
def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def load_object(filename):
    with open(filename, 'rb') as inp:
        return pickle.load(inp)


# Data error only
class CodeSimulator_DataError():
    def __init__(self, code=None, decoder_x=None, decoder_z=None, pauli_error_probs = [0.01, 0.01, 0.01], eval_logical_type = 'Total'):
        self.code = code
        self.decoder_z, self.decoder_x = decoder_z, decoder_x
        self.N = code.N
        self.K = code.K
        self.channel_probs = pauli_error_probs
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
        self.eval_logical_type = eval_logical_type # The type of the logical error to evaluate. X/Z/Total
    
    def _generate_error(self):

        '''
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        '''
        
        self.error_x = np.zeros(self.N).astype(int) 
        self.error_z = np.zeros(self.N).astype(int)

        for i in range(self.N):
#             rand = np.random.random()
            rand = random.random()
            if rand < self.channel_probs[2]:
                self.error_z[i] = 1
                self.error_x[i] = 0
            elif self.channel_probs[2] <= rand < (self.channel_probs[2]+self.channel_probs[0]):
                self.error_z[i] = 0
                self.error_x[i] = 1
            elif (self.channel_probs[2]+self.channel_probs[0]) <= rand < (self.channel_probs[2]+self.channel_probs[0]+self.channel_probs[1]):
                self.error_z[i] = 1
                self.error_x[i] = 1
            else:
                self.error_z[i] = 0
                self.error_x[i] = 0

        return self.error_x, self.error_z
    
    def _single_run(self):

        '''
        The main simulation procedure
        '''

        # randomly generate the error
        self.error_x, self.error_z = self._generate_error()
        
        # decode z
        synd_z = self.code.hx@self.error_z % 2
        decoded_z = self.decoder_z.decode(synd_z)

        # decode x
        synd_x = self.code.hz@self.error_x % 2
        decoded_x = self.decoder_x.decode(synd_x)

        #compute the logical and word error rates
        residual_x = (self.error_x+decoded_x) % 2
        residual_z = (self.error_z+decoded_z) % 2

        # check for X failure
        X_failure = 0
        # if residual_x.any():
        if ((self.code.hz@residual_x) % 2).any():
            X_failure = 1
            # None
        if ((self.code.lz@residual_x) % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            X_failure = 1

        # check for Z failure:
        Z_failure = 0
        # if residual_z.any():
        if ((self.code.hx@residual_z)% 2).any():
            Z_failure = 1
            # None
        if ((self.code.lx@residual_z) % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            Z_failure = 1

        assert self.eval_logical_type in ['X', 'Z', 'Total']
        if self.eval_logical_type == 'X':
            return X_failure
        elif self.eval_logical_type == 'Z':
            return Z_failure
        elif self.eval_logical_type == 'Total':
            return (X_failure or Z_failure)
        
    def WordErrorRate(self, num_run:int):
        eval_func = lambda physical_error_rate: self._single_run()        
        sim_if_error_list = parmap(eval_func, [0]*num_run, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)

        # compute logical error rate
        logical_error_rate = error_count/num_run
        logical_error_rate_eb = np.sqrt(
            (1-logical_error_rate)*logical_error_rate/num_run)

        # compute word error rate
        word_error_rate = 1.0 - \
            (1-logical_error_rate)**(1/self.K)

        word_error_rate_eb = logical_error_rate_eb * \
            ((1-logical_error_rate_eb)**(1/self.K - 1))/self.K
        
        return word_error_rate, word_error_rate_eb

    
    

# Phenomenological-level simulator
class CodeSimulator_Phenon():
    def __init__(self, code=None, decoder1_x=None, decoder1_z=None, decoder2_x=None, decoder2_z=None, pauli_error_probs = [0.01, 0.01, 0.01], q=0, eval_logical_type = 'Total'):
        self.code = code
        ## Modify the code to add new varaible nodes for syndrome errors
        self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
        self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
        self.decoder1_z, self.decoder1_x = decoder1_z, decoder1_x
        self.decoder2_z, self.decoder2_x = decoder2_z, decoder2_x
        self.N = code.N
        self.K = code.K
        self.channel_probs = pauli_error_probs
        self.synd_prob = q # The syndrom error prob equals 2/3 p for depolarizing noise
        
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
        self.eval_logical_type = eval_logical_type
    
    def _generate_error(self):

        '''
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        '''
        
#         self.error_x = np.zeros(self.N).astype(int) 
#         self.error_z = np.zeros(self.N).astype(int)
        
        self.error_x_ext = np.zeros(np.shape(self.hz_ext[1])).astype(int) 
        self.error_z_ext = np.zeros(np.shape(self.hx_ext[1])).astype(int)

        for i in range(self.N):
#             rand = np.random.random()
            rand = random.random()
            if rand < self.channel_probs[2]:
                self.error_z_ext[i] = 1
                self.error_x_ext[i] = 0
            elif self.channel_probs[2] <= rand < (self.channel_probs[2]+self.channel_probs[0]):
                self.error_z_ext[i] = 0
                self.error_x_ext[i] = 1
            elif (self.channel_probs[2]+self.channel_probs[0]) <= rand < (self.channel_probs[2]+self.channel_probs[0]+self.channel_probs[1]):
                self.error_z_ext[i] = 1
                self.error_x_ext[i] = 1
            else:
                self.error_z_ext[i] = 0
                self.error_x_ext[i] = 0
        
        ## Add syndrom error
        for i in range(np.shape(self.hx_ext)[1] - self.N):
            rand = random.random()
            if rand < self.synd_prob:
                self.error_z_ext[self.N + i] = 1
        
        for i in range(np.shape(self.hz_ext)[1] - self.N):
            rand = random.random()
            if rand < self.synd_prob:
                self.error_x_ext[self.N + i] = 1
            
        return self.error_x_ext, self.error_z_ext
    
    def _single_run(self, num_rounds):

        '''
        Run the noisey QEC num_rounds rounds using dec1, and apply a final perfect QEC round using dec2
        '''
        
        ## Simulate num_rounds rounds using dec1
        current_error_z_ext, current_error_x_ext = np.zeros(np.shape(self.hx_ext)[1]), np.zeros(np.shape(self.hz_ext)[1])
        for i in range(num_rounds - 1):
            # Generate new error for each round
            error_x_ext, error_z_ext = self._generate_error()
            current_error_x_ext = (np.hstack([current_error_x_ext[:self.N], np.zeros(np.shape(self.code.hz)[0])]) + error_x_ext) % 2
            current_error_z_ext = (np.hstack([current_error_z_ext[:self.N], np.zeros(np.shape(self.code.hx)[0])]) + error_z_ext) % 2
            
            # decode z
            synd_z = self.hx_ext@current_error_z_ext % 2
            decoded_z_ext = self.decoder1_z.decode(synd_z)

            # decode x
            synd_x = self.hz_ext@current_error_x_ext % 2
            decoded_x_ext = self.decoder1_x.decode(synd_x)
            
            # Calculate the residual error after correction
            current_error_x_ext = (current_error_x_ext + decoded_x_ext) % 2
            current_error_z_ext = (current_error_z_ext + decoded_z_ext) % 2

        ## Final round of perfect decoding use dec2
        error_x_ext, error_z_ext = self._generate_error()
        current_error_x = ((current_error_x_ext + error_x_ext)%2)[:self.N]
        current_error_z = ((current_error_z_ext + error_z_ext)%2)[:self.N]

        # current_error_x = ((current_error_x_ext + 0*error_x_ext)%2)[:self.N]
        # current_error_z = ((current_error_z_ext + 0*error_z_ext)%2)[:self.N]
        
        # decode z
        synd_z = self.code.hx@current_error_z % 2
        decoded_z = self.decoder2_z.decode(synd_z)

        # decode x
        synd_x = self.code.hz@current_error_x % 2
        decoded_x = self.decoder2_x.decode(synd_x)

        #compute the logical and word error rates
        residual_x = (current_error_x+decoded_x) % 2
        residual_z = (current_error_z+decoded_z) % 2
        
        # check for X failure
        X_failure = 0
        if (self.code.hz@residual_x % 2).any():
            X_failure = 1
        if (self.code.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            X_failure = 1

        # check for Z failure
        Z_failure = 0
        if (self.code.hx@residual_z % 2).any():
            Z_failure = 1
        elif (self.code.lx@residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            Z_failure = 1
        
        # elif ((self.code.hx@residual_z % 2).any()) or ((self.code.hz@residual_x % 2).any()):
        #     return 1
        
        assert self.eval_logical_type in ['X', 'Z', 'Total']
        if self.eval_logical_type == 'X':
            return X_failure
        elif self.eval_logical_type == 'Z':
            return Z_failure
        elif self.eval_logical_type == 'Total':
            return (X_failure or Z_failure)
        
    def WordErrorRate(self, num_rounds:int, num_samples:int):
        eval_func = lambda physical_error_rate: self._single_run(num_rounds)        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)

        # # compute logical error rate
        # logical_error_rate = error_count/num_samples
        # # logical_error_rate = 1.0 - (1-logical_error_rate)**(1/num_rounds)
        # logical_error_rate_per_cycle = (1.0 - (1-2*logical_error_rate)**(1/num_rounds))/2
        
        # logical_error_rate_eb = np.sqrt(
        #     (1-logical_error_rate_per_cycle)*logical_error_rate_per_cycle/num_samples)

        # # compute word error rate
        # word_error_rate = 1.0 - (1-logical_error_rate_per_cycle)**(1/self.K)
        # word_error_rate_eb = logical_error_rate_eb * \
        #     ((1-logical_error_rate_eb)**(1/self.K - 1))/self.K
        
        assert int(num_rounds)%2 == 1 # the number of cycles have to be odd in order to have invertible functons from wer to logical error rate
        logical_error_rate = error_count/num_samples # total logical error rate
        logical_error_rate_per_qubit = 1.0 - (1-logical_error_rate)**(1/self.K) # logical error rate per qubit

        if logical_error_rate_per_qubit <= 0.5:
            wer = (1.0 - (1-2*logical_error_rate_per_qubit)**(1/num_rounds))/2 # logical error rate per qubit per cycle
        else:
            wer = (1.0 + (-1+2*logical_error_rate_per_qubit)**(1/num_rounds))/2 # logical error rate per qubit per cycle

        return wer, None


    def WordErrorProbability(self, num_rounds:int, num_samples:int):
        eval_func = lambda physical_error_rate: self._single_run(num_rounds)        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)

        # compute logical error rate at the end
        logical_error_prob = error_count/num_samples
        
        logical_error_prob_eb = np.sqrt(
            (1-logical_error_prob)*logical_error_prob/num_samples)

        # compute word error rate
        word_error_prob = 1.0 - \
            (1-logical_error_prob)**(1/self.K)
        word_error_prob_eb = logical_error_prob_eb * \
            ((1-logical_error_prob_eb)**(1/self.K - 1))/self.K
        
        return word_error_prob, word_error_prob_eb

# Circuit-level simulator
class CodeSimulator_Circuit():
    def __init__(self, code=None, decoder1_z=None, decoder1_x=None, decoder2_z=None, decoder2_x=None, 
        p=0, num_cycles=1, error_params=None, eval_logical_type='Z', circuit_type = 'coloration', rand_scheduling_seed = 0):
         
        if eval_logical_type == 'X':
            # swap X and Z components of the code and the decoder

            temp = copy.deepcopy(code.hz)
            code.hz = code.hx
            code.hx = temp

            temp = copy.deepcopy(code.lz)
            code.lz = code.lx
            code.lx = temp

            decoder1_z = decoder1_x
            decoder2_z = decoder2_x


        self.eval_code = code
        ## Modify the code to add new varaible nodes for syndrome errors
        self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
        self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
        self.decoder1_z = decoder1_z
        self.decoder2_z = decoder2_z
        self.N = code.N
        self.K = code.K
        self.pz = p
        self.synd_prob = p
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
        self.num_cycles = num_cycles
        self.circuit = stim.Circuit()
        self.error_params = error_params
        
        self.max_stab_weight_x = int(np.max(np.sum(code.hx, axis = 1)))
        self.max_stab_weight_z = int(np.max(np.sum(code.hz, axis = 1)))
        
        # Obtain the circuit schedulings based on the circuit_type
        if circuit_type == 'random':
            self.scheduling_X = RandomCircuit(code.hx)
            self.scheduling_Z = RandomCircuit(code.hz)
                
        elif circuit_type == 'coloration':
            self.scheduling_X = ColorationCircuit(code.hx)
            self.scheduling_Z = ColorationCircuit(code.hz)
        
        
    def _generate_circuit(self):
        '''
        Generate a stim circuit
        '''
        # Set the circuit parameters
        hx = self.eval_code.hx
        hz = self.eval_code.hz
        lx = self.eval_code.lx
        data_indices = list(np.arange(0, np.shape(hx)[1]))
        n = len(data_indices)
        n_Z_ancilla, n_X_ancilla = np.shape(hz)[0], np.shape(hx)[0]
        Z_ancilla_indices = list(np.arange(n, n + n_Z_ancilla))
        X_ancilla_indices = list(np.arange(n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla))

        z_stab_weight = int(np.sum(hz[0,:]))
        x_stab_weight = int(np.sum(hx[0,:]))
        
        
        ## Initialization layer
        circuit_init = stim.Circuit()
        circuit_init.append("RX", data_indices)
        
        
        ## Initial stabilizer measurement layer
        circuit_stab_meas_init = stim.Circuit()
        #  measure the X stabilizers
        circuit_stab_meas_init.append("R", X_ancilla_indices)
        circuit_stab_meas_init.append("H", X_ancilla_indices)
        circuit_stab_meas_init.append("DEPOLARIZE1", X_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        circuit_stab_meas_init.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        circuit_stab_meas_init.append("TICK")
#        # Apply CX gates for the X stabilizers
        for time_step in range(len(self.scheduling_X)):
            idling_data_indices = list(copy.deepcopy(data_indices))

            for j in self.scheduling_X[time_step]:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                data_index = self.scheduling_X[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_init.append("CX", [X_ancilla_index, data_index])
                idling_data_indices.pop(idling_data_indices.index(data_index))
                
            circuit_stab_meas_init.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_init.append("TICK")

        
        # measure the Z stabilizers
        circuit_stab_meas_init.append("R", Z_ancilla_indices)
        circuit_stab_meas_init.append("DEPOLARIZE1", Z_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        circuit_stab_meas_init.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        circuit_stab_meas_init.append("TICK")
        # Appy CX gates for the Z stabilziers
        for time_step in range(len(self.scheduling_Z)):
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in self.scheduling_Z[time_step]:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                data_index = self.scheduling_Z[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas_init.append("CX", [data_index, Z_ancilla_index])
                idling_data_indices.pop(idling_data_indices.index(data_index))

            circuit_stab_meas_init.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas_init.append("TICK")

        # Measure the ancillas
        circuit_stab_meas_init.append("H", X_ancilla_indices)
        circuit_stab_meas_init.append("DEPOLARIZE1",  X_ancilla_indices, (self.error_params['p_m'])) # Add the measurement error
        circuit_stab_meas_init.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas_init.append("MR", Z_ancilla_indices + X_ancilla_indices)

        # Add the detector for X ancillas:
        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas_init.append("DETECTOR", stim.target_rec(- len(X_ancilla_indices) + i))

        circuit_stab_meas_init.append("TICK")
        
        
        ## Repeated stabilizer measurement layer
        circuit_stab_meas = stim.Circuit()
        # measurement the X ancillas
        # # Initialize the X ancillas to the + state
        circuit_stab_meas.append("H", X_ancilla_indices)
        circuit_stab_meas.append("DEPOLARIZE1", X_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        circuit_stab_meas.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for X ancillas
        circuit_stab_meas.append("TICK")
        # Apply CX gates for the X stabilizers
        for time_step in range(len(self.scheduling_X)):
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in self.scheduling_X[time_step]:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                data_index = self.scheduling_X[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas.append("CX", [X_ancilla_index, data_index])
                idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas.append("TICK")

        # meausure the Z ancillas
        ## initialize the Z ancillas
        circuit_stab_meas.append("DEPOLARIZE1", Z_ancilla_indices, (self.error_params['p_state_p'])) # Add the state preparation error
        circuit_stab_meas.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the preparation for Z ancillas
        circuit_stab_meas.append("TICK")
        # Appy CX gates for the Z stabilziers
        for time_step in range(len(self.scheduling_Z)):
            idling_data_indices = list(copy.deepcopy(data_indices))
            for j in self.scheduling_Z[time_step]:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                data_index = self.scheduling_Z[time_step][j]
                # data_index = supported_data_qubits[i]
                circuit_stab_meas.append("CX", [data_index, Z_ancilla_index])
                idling_data_indices.pop(idling_data_indices.index(data_index))
            circuit_stab_meas.append("DEPOLARIZE1", idling_data_indices, (self.error_params['p_i'])) # idling errors for qubits that are not being checked
            circuit_stab_meas.append("TICK")

        # Measure the ancillas
        circuit_stab_meas.append("H", X_ancilla_indices)
        circuit_stab_meas.append("DEPOLARIZE1",  X_ancilla_indices, (self.error_params['p_m'])) # Add the measurement error
        circuit_stab_meas.append("DEPOLARIZE1", data_indices, (self.error_params['p_i'])) # Add the idling errors on the data qubits during the measurement of X ancillas
        circuit_stab_meas.append("MR", Z_ancilla_indices + X_ancilla_indices)

        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                            stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))])
        circuit_stab_meas.append("TICK")

        
        ## Final projective measurement layer
        circuit_final_meas = stim.Circuit()
        circuit_final_meas.append("DEPOLARIZE1",  data_indices, (self.error_params['p_m'])) # Add the measurement error
        circuit_final_meas.append("MX", data_indices)

        # Obtain the syndroms
        for i in range(len(X_ancilla_indices)):
            supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - n - n_Z_ancilla,:] == 1)[0])

            rec_indices = []
            for data_index in supported_data_indices:
                rec_indices.append(- len(data_indices) + data_index)

            rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))

            circuit_final_meas.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices])

        # Obtain the logical measurements result
        for i in range(len(lx)):
            logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
            circuit_final_meas.append("OBSERVABLE_INCLUDE", 
                               [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
                               (i))
            
        circuit = circuit_init + circuit_stab_meas_init + (self.num_cycles - 2)*circuit_stab_meas + circuit_final_meas
        
        # Add noise
        noisy_circuit = circuit
        ## Add CX gate errors
        noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % self.error_params["p_CX"])

        # ## Add measurement errors
        # noisy_circuit = AddMeasurementError(noisy_circuit, meas_p=self.error_params["p_m"])
        
        # ## Add reset errors
        # noisy_circuit = AddResetError(noisy_circuit, reset_p=self.error_params["p_state_p"])
        
        # ## Add idling errors
        # noisy_circuit = AddIdlingError(noisy_circuit, error_instruction='Z_ERROR(%f)' % (self.error_params["p_i"]),
        #                               target_qubit_indices=data_indices)

        self.circuit = noisy_circuit


    def _decoding_samples(self, samples):
#         print('error params:', self.error_params)
        '''Decoding the samples obtained from the circuit_sampler'''
        syndrome_historys = [1.0*np.reshape(np.array(sample[:-len(self.eval_code.lx)]), [self.num_cycles, np.shape(self.eval_code.hx)[0]]) for sample in samples]
        logical_values = [1.0*np.array(sample[-len(self.eval_code.lx):]) for sample in samples]
        
#         print('syndrome history:', syndrome_historys, 'logical_values:', logical_values)
        if_failure_list = []

        for i in range(len(samples)):
            syndrome_history = syndrome_historys[i]
            logical_value = logical_values[i]

            correction = np.zeros(self.N)
            residual_syndrome = 0
            for j in range(len(syndrome_history) - 1):
                corrected_syndrome = (syndrome_history[j] + residual_syndrome)%2
                new_correction = self.decoder1_z.decode(corrected_syndrome)
                correction = (new_correction[:self.N] + correction)%2
#                 print('new corr:', np.shape(new_correction))
                residual_syndrome = (corrected_syndrome + self.eval_code.hx@new_correction[:self.N])%2

            corrected_syndrome_final = (syndrome_history[-1] + residual_syndrome)%2
            final_correction = self.decoder2_z.decode(corrected_syndrome_final)
            total_correction = (correction + final_correction)%2
            residual_syndrome_final = (corrected_syndrome_final + self.decoder2_z.h@final_correction)%2
            logical_correction = (self.eval_code.lx@total_correction)%2
            residual_logical_final = (logical_value + logical_correction)%2

            if_failure = residual_syndrome_final.any() or residual_logical_final.any()
            if_failure_list.append(1*if_failure)

        return if_failure_list
    
    def _single_run(self):
        detector_sampler = self.circuit.compile_detector_sampler()
        samples = detector_sampler.sample(shots=1, append_observables=True)
        if_failure = self._decoding_samples(samples)[0]
        
        return if_failure
    
    def WordErrorRate(self, num_samples:int):
        num_rounds = self.num_cycles
        assert int(num_rounds)%2 == 1 # the number of cycles have to be odd in order to have invertible functons from wer to logical error rate

        eval_func = lambda physical_error_rate: self._single_run()        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)
        # compute logical error rate

        logical_error_rate = error_count/num_samples # total logical error rate
        logical_error_rate_per_qubit = 1.0 - (1-logical_error_rate)**(1/self.K) # logical error rate per qubit

        if logical_error_rate_per_qubit <= 0.5:
            wer = (1.0 - (1-2*logical_error_rate_per_qubit)**(1/num_rounds))/2 # logical error rate per qubit per cycle
        else:
            wer = (1.0 + (-1+2*logical_error_rate_per_qubit)**(1/num_rounds))/2 # logical error rate per qubit per cycle
        
        return wer, None


# Functions for fitting the threshold
def CriticalExponentFit(xdata_tuple, pc, nu, A, B, C):
    p, d = xdata_tuple
    x = (p - pc)*d**(1/nu)
    pl = A + B*x + C*x**2
    return pl

def EmpericalFit(xdata_tuple, pc, A):
    p, d = xdata_tuple
    pl = A*(p/pc)**(d/2)
    return pl

def FitDistance(p, A, d):
    pl = A*p**(d/2)
    return pl

def DistanceEst(sweep_p_list, sweep_pl_total_list, if_plot=False):
    num_p = len(sweep_p_list)
    num_code = len(sweep_pl_total_list)
    sweep_d_list = []
    for sweep_pl_list in sweep_pl_total_list:
        initial_guess = (0.01, 3)
        popt, pcov = curve_fit(FitDistance, np.array(sweep_p_list), np.array(sweep_pl_list) + 1e-10, p0=initial_guess)
        sweep_d_list.append(popt[1])

    return sweep_d_list

def ThresholdEst_extrapolation(sweep_p_list, sweep_pl_total_list, if_plot=False):
    num_p = len(sweep_p_list)
    num_code = len(sweep_pl_total_list)
    sweep_d_list = DistanceEst(sweep_p_list, sweep_pl_total_list, if_plot=False)
    
    fit_d_list = copy.deepcopy(sweep_d_list)
    sweep_p_list = list(sweep_p_list)*num_code
    sweep_d1_list = []
    for sweep_d in sweep_d_list:
        sweep_d1_list += [sweep_d]*num_p
    sweep_d_list = sweep_d1_list
    sweep_pl_list = list(np.reshape(np.array(sweep_pl_total_list) + 1e-10, [num_p*num_code, ]))
    
    fit_X = np.vstack([np.reshape(np.array(sweep_p_list), [1, num_p*num_code]), 
                       np.reshape(np.array(sweep_d_list), [1, num_p*num_code])])
    fit_Z = np.reshape(np.array(sweep_pl_total_list), [num_p*num_code, ])
    initial_guess = (0.04, 0.1)
    popt, pcov = curve_fit(EmpericalFit, fit_X, fit_Z, p0=initial_guess)
    perr = np.sqrt(np.diag(pcov))
    
    # plot
    p_c, A = popt[0], popt[1]
    fit_p_list = list(set(sweep_p_list))
    fit_pl_list = np.reshape(np.array(sweep_pl_list), [len(fit_d_list), len(fit_p_list)])
    if if_plot:
        fitted_pl_list = []
        for sweep_d in fit_d_list:
            fitted_pl_list.append([EmpericalFit((sweep_p, sweep_d), p_c, A) for sweep_p in fit_p_list])
        
        plt.figure()
        for i in range(len(fit_d_list)):
            plt.plot(fit_p_list, fitted_pl_list[i], '-', c = 'C%i'%i)
            plt.plot(sweep_p_list[:num_p], sweep_pl_list[i*num_p:(i + 1)*num_p], 'D', c = 'C%i'%i)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('p')
        plt.ylabel('WER')
    
    print('p_c:', popt[0])
    
    return popt[0]



# Code family
class CodeFamily():
    def __init__(self, code_list:list, decoder1_class:DecoderClass, decoder2_class:DecoderClass):
        self.code_list = code_list
        self.decoder1_class = decoder1_class
        self.decoder2_class = decoder2_class
    
    def EvalWER(self, noise_model:str, eval_logical_type:str, eval_p_list:list, 
                      num_samples:int, num_cycles=1, data_synd_noise_ratio=1, circuit_type='coloration',
                circuit_error_params=None, if_plot=True):
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        
        # Data threshold
        if noise_model == 'data':
            eval_wer_list = []
            for eval_code in self.code_list:
                for eval_p in eval_p_list:
                    p = eval_p*3/2
                    pauli_error_probs = [p/3, p/3, p/3] # depolarizing channel [px, py, pyz]
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': eval_p}
                    decoder_x = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': eval_p}
                    decoder_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                
                
                    code_simulator = CodeSimulator_DataError(code=eval_code, decoder_x=decoder_x, 
                                      decoder_z=decoder_z, pauli_error_probs=pauli_error_probs,
                                                             eval_logical_type=eval_logical_type)

                    eval_wer_list.append(code_simulator.WordErrorRate(num_samples)[0])
        
        # Phenomenological threshold
        elif noise_model == 'phenl':
            eval_wer_list = []
            for eval_code in self.code_list:
                for eval_p in eval_p_list:
                    p = 3/2*eval_p
                    q = eval_p

                    p_data = p*2/3
                    p_synd = q

                    pauli_error_probs = [p/3, p/3, p/3] # depolarizing channel [px, py, pyz]

                    eval_hx_extended = np.hstack([eval_code.hx, np.identity(np.shape(eval_code.hx)[0])])
                    eval_hz_extended = np.hstack([eval_code.hz, np.identity(np.shape(eval_code.hz)[0])])
                    
                    code_and_noise_channel_params = {'h': eval_hz_extended, 'p_data': p_data, 'p_syndrome':p_synd}
                    dec1_x = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_hx_extended, 'p_data': p_data, 'p_syndrome':p_synd}
                    dec1_z = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': p_data}
                    dec2_x = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': p_data}
                    dec2_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_simulator = CodeSimulator_Phenon(code=eval_code, decoder1_x= dec1_x, decoder1_z= dec1_z,
                                                              decoder2_x=dec2_x, decoder2_z=dec2_z,
                                                              pauli_error_probs=pauli_error_probs, q=q, eval_logical_type=eval_logical_type)

                    eval_wer_list.append(code_simulator.WordErrorRate(num_rounds=num_cycles, num_samples=num_samples)[0])

        
        # Circuit threshold
        elif noise_model == 'circuit':
            eval_wer_list = []
            for eval_code in self.code_list:
                for eval_p in eval_p_list:
                    p = eval_p
                    error_params = {"p_i": circuit_error_params["p_i"]*p, "p_state_p": circuit_error_params["p_state_p"]*p, "p_m": circuit_error_params["p_m"]*p, "p_CX":circuit_error_params["p_CX"]*p, 
                    "p_idling_gate": circuit_error_params["p_idling_gate"]*p}

                    # print('error parameters:', error_params)

                    p_data = data_synd_noise_ratio*p 
                    p_synd = 1*p

                    eval_hx_ext = np.hstack([eval_code.hx, np.identity(np.shape(eval_code.hx)[0])])
                    eval_hz_ext = np.hstack([eval_code.hz, np.identity(np.shape(eval_code.hz)[0])])

                    code_and_noise_channel_params = {'h': eval_hx_ext, 'p_data': p_data, 'p_syndrome':p_synd}
                    dec1_z = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_hz_ext, 'p_data': p_data, 'p_syndrome':p_synd}
                    dec1_x = self.decoder1_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hx, 'p_data': eval_p}
                    dec2_z = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    code_and_noise_channel_params = {'h': eval_code.hz, 'p_data': eval_p}
                    dec2_x = self.decoder2_class.GetDecoder(code_and_noise_channel_params)
                    
                    if eval_logical_type == 'Total':
#                         print('evaluating total wer')
                        # approximate the total logical error rate as px + pz
                        code_simulator_circuit_z = CodeSimulator_Circuit(code=eval_code, decoder1_z= dec1_z, decoder1_x= dec1_x,
                                                          decoder2_z=dec2_z, decoder2_x=dec2_x, p=p, num_cycles=num_cycles, 
                                                          error_params=error_params, eval_logical_type='Z', 
                                                                   circuit_type = circuit_type, rand_scheduling_seed=1)
                        code_simulator_circuit_z._generate_circuit()
                        wer_z = code_simulator_circuit_z.WordErrorRate(num_samples=num_samples)[0]
                        
                        code_simulator_circuit_x = CodeSimulator_Circuit(code=eval_code, decoder1_z= dec1_z, decoder1_x= dec1_x,
                                                          decoder2_z=dec2_z, decoder2_x=dec2_x, p=p, num_cycles=num_cycles, 
                                                          error_params=error_params, eval_logical_type='X', 
                                                                   circuit_type = circuit_type, rand_scheduling_seed=1)
                        code_simulator_circuit_x._generate_circuit()
                        wer_x = code_simulator_circuit_x.WordErrorRate(num_samples=num_samples)[0]
                        
                        wer = wer_z + wer_x
                        eval_wer_list.append(wer)
                    
                    else:
                        code_simulator_circuit = CodeSimulator_Circuit(code=eval_code, decoder1_z= dec1_z, decoder1_x= dec1_x,
                                                                       decoder2_z=dec2_z, decoder2_x=dec2_x, p=p, num_cycles=num_cycles, 
                                                                       error_params=error_params, eval_logical_type=eval_logical_type, 
                                                                       circuit_type = circuit_type, rand_scheduling_seed=1)
                        
                        code_simulator_circuit._generate_circuit()
                        eval_wer_list.append(code_simulator_circuit.WordErrorRate(num_samples=num_samples)[0])
            
        else:
            print('no valid error model selected')
                
        eval_wer_array = np.reshape(np.array(eval_wer_list), [len(self.code_list), len(eval_p_list)])
            
        if if_plot == True:
            logical_error_per_qubit_array = (1 - (1 - 2*eval_wer_array)**num_cycles)/2
            logical_error_array = np.zeros(eval_wer_array.shape)
            for i in range(eval_wer_array.shape[0]):
                K = self.code_list[i].K
                logical_error_array[i,:] = 1 - (1 - logical_error_per_qubit_array[i,:])**K

            fig, ax = plt.subplots(1,3, figsize = (15,3))
            for eval_error in logical_error_array:
                ax[0].plot(eval_p_list, eval_error, 'D--')
            ax[0].set_xscale('log')
            ax[0].set_yscale('log')
            ax[0].set_xlabel(r'$p$')
            ax[0].set_ylabel('Logical error')
            
            for eval_error in logical_error_per_qubit_array:
                ax[1].plot(eval_p_list, eval_error, 'D--')
            ax[1].set_xscale('log')
            ax[1].set_yscale('log')
            ax[1].set_xlabel(r'$p$')
            ax[1].set_ylabel('Logical error per qubit')

            for eval_error in eval_wer_array:
                ax[2].plot(eval_p_list, eval_error, 'D--')
            ax[2].set_xscale('log')
            ax[2].set_yscale('log')
            ax[2].set_xlabel(r'$p$')
            ax[2].set_ylabel('WER')
                
            plt.show(fig)

        return eval_wer_array
        
        
        
    def EvalThreshold(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float,
                      num_samples:int, num_cycles=1, data_synd_noise_ratio=1, circuit_type='coloration', circuit_error_params=None, if_plot=False):
        """Evalutate the threshold of the code family"""
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        assert eval_method in ['extrapolation'], f'eval_method should be one of [extrapolation]'
        
        if eval_method == 'extrapolation':
            eval_p_list = 10**(np.linspace(np.log10(est_threshold*0.4), np.log10(est_threshold*0.8), 6))
            eval_wer_array = self.EvalWER(noise_model, eval_logical_type, eval_p_list, 
                      num_samples, num_cycles, data_synd_noise_ratio, circuit_type, circuit_error_params, if_plot=False)
            
            return ThresholdEst_extrapolation(eval_p_list, eval_wer_array, if_plot)
        
        
    def EvalSustainableThreshold(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float, 
                      num_samples_per_cycle:int, num_cycles_list:list, data_synd_noise_ratio=1, circuit_type='coloration', circuit_error_params=None, if_plot=False):
        
        sweep_threshold_list = [self.EvalThreshold(noise_model=noise_model, eval_logical_type=eval_logical_type, eval_method=eval_method, est_threshold=est_threshold, 
                                                   num_samples=int(num_samples_per_cycle/sweep_num_of_cycle), 
                                                   num_cycles=sweep_num_of_cycle, data_synd_noise_ratio=data_synd_noise_ratio, circuit_type=circuit_type, 
                                                   circuit_error_params=circuit_error_params, if_plot=if_plot) for sweep_num_of_cycle in num_cycles_list]
        
        # Fit the sustaimable threshold
        def FitSusThreshold(N, p_sus, p_0, gamma):
            pth = p_sus*(1 - (1 - p_0/p_sus)*np.exp(-gamma*N))
            return pth
        
        initial_guess = (0.01, 0.05, 0.05)
        popt, pcov = curve_fit(FitSusThreshold, np.array(num_cycles_list), np.array(sweep_threshold_list), p0=initial_guess)
        p_sus = popt[0]
        
        plt.figure()
        plt.plot(num_cycles_list, sweep_threshold_list, 'D')
        plt.plot(num_cycles_list, FitSusThreshold(np.array(num_cycles_list), popt[0], popt[1], popt[2]), '-')
        
        return p_sus
        
        
    def EvalEffectiveDistances(self, noise_model:str, eval_logical_type:str, eval_method:str, est_threshold:float,
                      num_samples:int, num_cycles=1, data_synd_noise_ratio=1, circuit_type='coloration', if_plot=False):
        """Evalutate the effective distances of the code family"""
        assert noise_model in ['data', 'phenl', 'circuit'], f'noise_model should be one of [data, phenl, circuit]'
        assert eval_logical_type in ['X', 'Z', 'Total'], f'eval_type should be one of [X, Y, Total]'
        assert eval_method in ['extrapolation'], f'eval_method should be one of [extrapolation]'
        
        
        eval_p_list = 10**(np.linspace(np.log10(est_threshold/6), np.log10(est_threshold/4), 5))
        eval_wer_array = self.EvalWER(noise_model, eval_logical_type, eval_p_list, 
                    num_samples, num_cycles, data_synd_noise_ratio, circuit_type, if_plot=False)
            
        return DistanceEst(eval_p_list, eval_wer_array, if_plot)
