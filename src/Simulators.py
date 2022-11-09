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
sys.path.append("/Users/qian/Box Sync/Research/Jiang_group_projects/QLDPC/HGP_Simulations/src/")
from Decoders import BPOSD_Decoder, FirstMinBPDecoder

sys.path.append("/Users/qian/Box Sync/Research/Jiang_group_projects/Squeezed_Cats_Synchronized/Squeezed_Cat/Codes/src/")
from ErrorPlugin import *


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
    def __init__(self, code=None, decoder_x=None, decoder_z=None, pauli_error_probs = [0.01, 0.01, 0.01]):
        self.code = code
        self.decoder_z, self.decoder_x = decoder_z, decoder_x
        self.N = code.N
        self.K = code.K
        self.channel_probs = pauli_error_probs
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
    
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

        # check for logical X-error
        if (self.code.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            return 1

        # check for logical Z-error
        elif (self.code.lx@residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            return 1
        else:
            return 0
        
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
    def __init__(self, code=None, decoder1_x=None, decoder1_z=None, decoder2_x=None, decoder2_z=None, pauli_error_probs = [0.01, 0.01, 0.01]):
        self.code = code
        ## Modify the code to add new varaible nodes for syndrome errors
        self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
        self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
        self.decoder1_z, self.decoder1_x = decoder1_z, decoder1_x
        self.decoder2_z, self.decoder2_x = decoder2_z, decoder2_x
        self.N = code.N
        self.K = code.K
        self.channel_probs = pauli_error_probs
        self.synd_prob = np.sum(pauli_error_probs)*2/3 # The syndrom error prob equals 2/3 p for depolarizing noise
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
    
    def _generate_error(self):

        '''
        Generates a random error on both the X and Z components of the code
        distributed according to the channel probability vectors.
        '''
        
#         self.error_x = np.zeros(self.N).astype(int) 
#         self.error_z = np.zeros(self.N).astype(int)
        
        self.error_x_ext = np.zeros(np.shape(self.hx_ext[1])).astype(int) 
        self.error_z_ext = np.zeros(np.shape(self.hz_ext[1])).astype(int)

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
        for i in range(num_rounds):
            # Generate new error for each round
            error_x_ext, error_z_ext = self._generate_error()
            current_error_x_ext = (np.hstack([current_error_x_ext[:self.N], np.zeros(np.shape(self.code.hz)[0])]) + error_x_ext) % 2
            current_error_z_ext = (np.hstack([current_error_z_ext[:self.N], np.zeros(np.shape(self.code.hz)[0])]) + error_z_ext) % 2
            
            # decode z
            synd_z = self.hx_ext@current_error_z_ext % 2
            decoded_z_ext = self.decoder1_z.decode(synd_z)

            # decode x
            synd_x = self.hz_ext@current_error_x_ext % 2
            decoded_x_ext = self.decoder1_x.decode(synd_x)
            
            # Calculate the residual error after correction
            current_error_x_ext = (current_error_x_ext + decoded_x_ext) % 2
            current_error_z_ext = (current_error_x_ext + decoded_z_ext) % 2

        ## Final round of perfect decoding use dec2
        current_error_x = current_error_x_ext[:self.N]
        current_error_z = current_error_z_ext[:self.N]
        
        # decode z
        synd_z = self.code.hx@current_error_z % 2
        decoded_z = self.decoder2_z.decode(synd_z)

        # decode x
        synd_x = self.code.hz@current_error_x % 2
        decoded_x = self.decoder2_x.decode(synd_x)

        #compute the logical and word error rates
        residual_x = (current_error_x+decoded_x) % 2
        residual_z = (current_error_z+decoded_z) % 2
        
        # check for logical X-error
        if (self.code.lz@residual_x % 2).any():
            logical_weight = np.sum(residual_x)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            return 1

        # check for logical Z-error
        elif (self.code.lx@residual_z % 2).any():
            logical_weight = np.sum(residual_z)
            if logical_weight < self.min_logical_weight:
                self.min_logical_weight = int(logical_weight)
            return 1
        
        elif ((self.code.hx@residual_z % 2).any()) or ((self.code.hz@residual_x % 2).any()):
            return 1
        
        else:
            return 0
        
    def WordErrorRate(self, num_rounds:int, num_samples:int):
        eval_func = lambda physical_error_rate: self._single_run(num_rounds)        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)

        # compute logical error rate
        logical_error_rate = error_count/num_samples
        logical_error_rate = 1.0 - (1-logical_error_rate)**(1/num_rounds)
        
        logical_error_rate_eb = np.sqrt(
            (1-logical_error_rate)*logical_error_rate/num_samples)

        # compute word error rate
        word_error_rate = 1.0 - \
            (1-logical_error_rate)**(1/self.K)
        word_error_rate_eb = logical_error_rate_eb * \
            ((1-logical_error_rate_eb)**(1/self.K - 1))/self.K
        
        return word_error_rate, word_error_rate_eb


    def WordErrorProbability(self, num_rounds:int, num_samples:int):
        eval_func = lambda physical_error_rate: self._single_run(num_rounds)        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
        error_count = np.sum(sim_if_error_list)

        # compute logical error rate
        logical_error_prob = error_count/num_samples
        
        logical_error_prob_eb = np.sqrt(
            (1-logical_error_prob)*logical_error_prob/num_samples)

        # compute word error rate
        word_error_prob = 1.0 - \
            (1-logical_error_prob)**(1/self.K)
        word_error_prob_eb = logical_error_prob_eb * \
            ((1-logical_error_prob_eb)**(1/self.K - 1))/self.K
        
        return word_error_prob, word_error_prob_eb


# # Circuit-level simulator
# class CodeSimulator_Circuit():
#     def __init__(self, code=None, decoder1_z=None, decoder2_z=None, pz=0, num_cycles=1, error_params=None):
#         self.eval_code = code
#         ## Modify the code to add new varaible nodes for syndrome errors
#         self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
#         self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
#         self.decoder1_z = decoder1_z
#         self.decoder2_z = decoder2_z
#         self.N = code.N
#         self.K = code.K
#         self.pz = pz
#         self.synd_prob = pz 
        
#         self.error_x = np.zeros(self.N).astype(int) #x_component error vector
#         self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
#         self.min_logical_weight = self.N
#         self.num_cycles = num_cycles
#         self.circuit = stim.Circuit()
#         self.error_params = error_params
        
        
#     def _generate_circuit(self):
#         '''
#         Generate a stim circuit
#         '''
#         # Set the circuit parameters
#         hx = self.eval_code.hx
#         hz = self.eval_code.hz
#         lx = self.eval_code.lx
#         data_indices = list(np.arange(0, np.shape(hx)[1]))
#         n = len(data_indices)
#         n_Z_ancilla, n_X_ancilla = np.shape(hz)[0], np.shape(hx)[0]
#         Z_ancilla_indices = list(np.arange(n, n + n_Z_ancilla))
#         X_ancilla_indices = list(np.arange(n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla))

#         z_stab_weight = int(np.sum(hz[0,:]))
#         x_stab_weight = int(np.sum(hx[0,:]))
        
        
#         ## Initialization layer
#         circuit_init = stim.Circuit()
#         circuit_init.append("RX", data_indices)
        
        
#         ## Initial stabilizer measurement layer
#         circuit_stab_meas_init = stim.Circuit()
#         circuit_stab_meas_init.append("R", Z_ancilla_indices + X_ancilla_indices)
#         # # Initialize the X ancillas to the + state
#         circuit_stab_meas_init.append("H", X_ancilla_indices)

#         circuit_stab_meas_init.append("TICK")

#         # Apply CX gates for the X stabilizers
#         for i in range(x_stab_weight):
#             for X_ancilla_index in X_ancilla_indices:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
#                 data_index = supported_data_qubits[i]
#                 circuit_stab_meas_init.append("CX", [X_ancilla_index, data_index])
#             circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + X_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
#             circuit_stab_meas_init.append("TICK")
            
#         # Appy CX gates for the Z stabilziers
#         for i in range(z_stab_weight):
#             for Z_ancilla_index in Z_ancilla_indices:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
#                 data_index = supported_data_qubits[i]
#                 circuit_stab_meas_init.append("CX", [data_index, Z_ancilla_index])
#             circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + Z_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
#             circuit_stab_meas_init.append("TICK")

#         # Measure the ancillas
#         circuit_stab_meas_init.append("H", X_ancilla_indices)
#         circuit_stab_meas_init.append("MR", Z_ancilla_indices + X_ancilla_indices)

#         # Add the detector for X ancillas:
#         for i in range(len(X_ancilla_indices)):
#             circuit_stab_meas_init.append("DETECTOR", stim.target_rec(- len(X_ancilla_indices) + i))

#         circuit_stab_meas_init.append("TICK")
        
        
#         ## Repeated stabilizer measurement layer
#         circuit_stab_meas = stim.Circuit()
#         # Intialize the ancillas
# #         circuit_stab_meas.append("R", Z_ancilla_indices + X_ancilla_indices)
#         # # Initialize the X ancillas to the + state
#         circuit_stab_meas.append("H", X_ancilla_indices)

#         circuit_stab_meas.append("TICK")

#         # Apply CX gates for the X stabilizers
#         for i in range(x_stab_weight):
#             for X_ancilla_index in X_ancilla_indices:
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
#                 data_index = supported_data_qubits[i]
#                 circuit_stab_meas.append("CX", [X_ancilla_index, data_index])
#             circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + X_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
#             circuit_stab_meas.append("TICK")
#         # Appy CX gates for the Z stabilziers
#         for i in range(z_stab_weight):
#             for Z_ancilla_index in Z_ancilla_indices:
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
#                 data_index = supported_data_qubits[i]
#                 circuit_stab_meas.append("CX", [data_index, Z_ancilla_index])
#             circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + Z_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
#             circuit_stab_meas.append("TICK")

#         # Measure the ancillas
#         circuit_stab_meas.append("H", X_ancilla_indices)
#         circuit_stab_meas.append("MR", Z_ancilla_indices + X_ancilla_indices)

#         for i in range(len(X_ancilla_indices)):
#             circuit_stab_meas.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
#                                             stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))])
#         circuit_stab_meas.append("TICK")

        
#         ## Final projective measurement layer
#         circuit_final_meas = stim.Circuit()
#         circuit_final_meas.append("MX", data_indices)

#         # Obtain the syndroms
#         for i in range(len(X_ancilla_indices)):
#             supported_data_indices = list(np.where(hx[X_ancilla_indices[i] - n - n_Z_ancilla,:] == 1)[0])

#             rec_indices = []
#             for data_index in supported_data_indices:
#                 rec_indices.append(- len(data_indices) + data_index)

#             rec_indices.append(- len(X_ancilla_indices) + i - len(data_indices))

#             circuit_final_meas.append("Detector", [stim.target_rec(rec_index) for rec_index in rec_indices])

#         # Obtain the logical measurements result
#         for i in range(len(lx)):
#             logical_X_qubit_indices = list(np.where(lx[i,:] == 1)[0])
#             circuit_final_meas.append("OBSERVABLE_INCLUDE", 
#                                [stim.target_rec(- len(data_indices) + data_index) for data_index in logical_X_qubit_indices],
#                                (i))
            
#         circuit = circuit_init + circuit_stab_meas_init + (self.num_cycles - 1)*circuit_stab_meas + circuit_final_meas
        
#         # Add noise
#         noisy_circuit = circuit
#         ## Add CX gate errors
#         noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % self.error_params["p_CX"])

#         ## Add measurement errors
#         noisy_circuit = AddMeasurementError(noisy_circuit, meas_p=self.error_params["p_m"])
        
#         ## Add reset errors
#         noisy_circuit = AddResetError(noisy_circuit, reset_p=self.error_params["p_state_p"])
        
#         ## Add idling errors
#         noisy_circuit = AddIdlingError(noisy_circuit, error_instruction='Z_ERROR(%f)' % (self.error_params["p_i"]),
#                                       target_qubit_indices=data_indices)

#         self.circuit = noisy_circuit

    
#     def _decoding_samples(self, samples):
#         '''Decoding the samples obtained from the circuit_sampler'''
#         syndrome_historys = [1.0*np.reshape(np.array(sample[:-len(self.eval_code.lx)]), [self.num_cycles + 1, np.shape(self.eval_code.hx)[0]]) for sample in samples]
#         logical_values = [1.0*np.array(sample[-len(self.eval_code.lx):]) for sample in samples]
        
#         if_failure_list = []

#         for i in range(len(samples)):
#             syndrome_history = syndrome_historys[i]
#             logical_value = logical_values[i]

#             correction = np.zeros(self.N)
#             residual_syndrome = 0
#             for j in range(len(syndrome_history) - 1):
#                 corrected_syndrome = (syndrome_history[j] + residual_syndrome)%2
#                 new_correction = self.decoder1_z.decode(corrected_syndrome)
#                 correction = (new_correction[:self.N] + correction)%2
# #                 print('new corr:', np.shape(new_correction))
#                 residual_syndrome = (corrected_syndrome + self.eval_code.hx@new_correction[:self.N])%2

#             corrected_syndrome_final = (syndrome_history[-1] + residual_syndrome)%2
#             final_correction = self.decoder2_z.decode(corrected_syndrome_final)
#             total_correction = (correction + final_correction)%2
#             residual_syndrome_final = (corrected_syndrome_final + self.decoder2_z.h@final_correction)%2
#             logical_correction = (self.eval_code.lx@total_correction)%2
#             residual_logical_final = (logical_value + logical_correction)%2

#             if_failure = residual_syndrome_final.any() or residual_logical_final.any()
#             if_failure_list.append(1*if_failure)

#         return if_failure_list
    
#     def _single_run(self):
#         detector_sampler = self.circuit.compile_detector_sampler()
#         samples = detector_sampler.sample(shots=1, append_observables=True)
#         if_failure = self._decoding_samples(samples)[0]
        
#         return if_failure
    
#     def WordErrorRate(self, num_samples:int):
#         num_rounds = self.num_cycles
#         eval_func = lambda physical_error_rate: self._single_run()        
#         sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
# #         error_count = np.sum(sim_if_error_list)/num_rounds
#         error_count = np.sum(sim_if_error_list)
#         # compute logical error rate
#         logical_error_rate = error_count/num_samples
        
#         logical_error_rate = 1.0 - (1-logical_error_rate)**(1/num_rounds)
        
#         logical_error_rate_eb = np.sqrt(
#             (1-logical_error_rate)*logical_error_rate/num_samples)

#         # compute word error rate
#         word_error_rate = 1.0 - \
#             (1-logical_error_rate)**(1/self.K)
#         word_error_rate_eb = logical_error_rate_eb * \
#             ((1-logical_error_rate_eb)**(1/self.K - 1))/self.K
        
#         return word_error_rate, word_error_rate_eb



# Circuit-level simulator
class CodeSimulator_Circuit():
    def __init__(self, code=None, decoder1_z=None, decoder2_z=None, pz=0, num_cycles=1, error_params=None, rand_scheduling_seed = 0):
        self.eval_code = code
        ## Modify the code to add new varaible nodes for syndrome errors
        self.hx_ext = np.hstack([code.hx, np.identity(np.shape(code.hx)[0])])
        self.hz_ext = np.hstack([code.hz, np.identity(np.shape(code.hz)[0])])
        
        self.decoder1_z = decoder1_z
        self.decoder2_z = decoder2_z
        self.N = code.N
        self.K = code.K
        self.pz = pz
        self.synd_prob = pz 
        
        self.error_x = np.zeros(self.N).astype(int) #x_component error vector
        self.error_z = np.zeros(self.N).astype(int) #z_component error vector
        
        self.min_logical_weight = self.N
        self.num_cycles = num_cycles
        self.circuit = stim.Circuit()
        self.error_params = error_params
        
        self.max_stab_weight_x = int(np.max(np.sum(code.hx, axis = 1)))
        self.max_stab_weight_z = int(np.max(np.sum(code.hz, axis = 1)))
        
        # Obtain a random scheduling 
        n = self.N
        n_Z_ancilla, n_X_ancilla = np.shape(code.hz)[0], np.shape(code.hx)[0]
        Z_ancilla_indices = list(np.arange(n, n + n_Z_ancilla))
        X_ancilla_indices = list(np.arange(n + n_Z_ancilla, n + n_Z_ancilla + n_X_ancilla))
        self.scheduling_X = [list(np.where(code.hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0]) for X_ancilla_index in X_ancilla_indices]
        self.scheduling_Z = [list(np.where(code.hz[Z_ancilla_index - n,:] == 1)[0]) for Z_ancilla_index in Z_ancilla_indices]
#         [random.shuffle(s) for s in self.scheduling_X]
#         [random.shuffle(s) for s in self.scheduling_Z]
#         rand_scheduling_seed = 0
        [random.Random(i + rand_scheduling_seed).shuffle(self.scheduling_X[i]) for i in range(len(self.scheduling_X))]
        [random.Random(i + rand_scheduling_seed).shuffle(self.scheduling_Z[i]) for i in range(len(self.scheduling_Z))]
        
        
        
        
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
        circuit_stab_meas_init.append("R", Z_ancilla_indices + X_ancilla_indices)
        # # Initialize the X ancillas to the + state
        circuit_stab_meas_init.append("H", X_ancilla_indices)

        circuit_stab_meas_init.append("TICK")

        # Apply CX gates for the X stabilizers
        for i in range(self.max_stab_weight_x):
#             for X_ancilla_index in X_ancilla_indices:
#             for X_ancilla_support_list in self.scheduling_X:
            for j in range(len(self.scheduling_X)):
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                supported_data_qubits = self.scheduling_X[j]
                data_index = supported_data_qubits[i]
                circuit_stab_meas_init.append("CX", [X_ancilla_index, data_index])
            circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + X_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
            circuit_stab_meas_init.append("TICK")
            
        # Appy CX gates for the Z stabilziers
        for i in range(z_stab_weight):
#             for Z_ancilla_index in Z_ancilla_indices:
#             for Z_ancilla_support_list in self.scheduling_Z:
            for j in range(len(self.scheduling_Z)):
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                supported_data_qubits = self.scheduling_Z[j]
        
                data_index = supported_data_qubits[i]
                circuit_stab_meas_init.append("CX", [data_index, Z_ancilla_index])
            circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + Z_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
            circuit_stab_meas_init.append("TICK")

        # Measure the ancillas
        circuit_stab_meas_init.append("H", X_ancilla_indices)
        circuit_stab_meas_init.append("MR", Z_ancilla_indices + X_ancilla_indices)

        # Add the detector for X ancillas:
        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas_init.append("DETECTOR", stim.target_rec(- len(X_ancilla_indices) + i))

        circuit_stab_meas_init.append("TICK")
        
        
        ## Repeated stabilizer measurement layer
        circuit_stab_meas = stim.Circuit()
        # Intialize the ancillas
#         circuit_stab_meas.append("R", Z_ancilla_indices + X_ancilla_indices)
        # # Initialize the X ancillas to the + state
        circuit_stab_meas.append("H", X_ancilla_indices)

        circuit_stab_meas.append("TICK")

        # Apply CX gates for the X stabilizers
        for i in range(x_stab_weight):
#             for X_ancilla_index in X_ancilla_indices:
            for j in range(len(self.scheduling_X)):
#                 supported_data_qubits = list(np.where(hx[X_ancilla_index - n - n_Z_ancilla,:] == 1)[0])
                X_ancilla_index = X_ancilla_indices[j]
                supported_data_qubits = self.scheduling_X[j]
                data_index = supported_data_qubits[i]
                circuit_stab_meas.append("CX", [X_ancilla_index, data_index])
            circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + X_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
            circuit_stab_meas.append("TICK")
        # Appy CX gates for the Z stabilziers
        for i in range(z_stab_weight):
#             for Z_ancilla_index in Z_ancilla_indices:
            for j in range(len(self.scheduling_X)):
#                 supported_data_qubits = list(np.where(hz[Z_ancilla_index - n,:] == 1)[0])
                Z_ancilla_index = Z_ancilla_indices[j]
                supported_data_qubits = self.scheduling_Z[j]
                data_index = supported_data_qubits[i]
                circuit_stab_meas.append("CX", [data_index, Z_ancilla_index])
            circuit_stab_meas_init.append("DEPOLARIZE1", data_indices + Z_ancilla_indices, (0.5*self.error_params['p_idling_gate']))
            circuit_stab_meas.append("TICK")

        # Measure the ancillas
        circuit_stab_meas.append("H", X_ancilla_indices)
        circuit_stab_meas.append("MR", Z_ancilla_indices + X_ancilla_indices)

        for i in range(len(X_ancilla_indices)):
            circuit_stab_meas.append("DETECTOR", [stim.target_rec(- len(X_ancilla_indices) + i), 
                                            stim.target_rec(- len(X_ancilla_indices) + i - len(Z_ancilla_indices) - len(X_ancilla_indices))])
        circuit_stab_meas.append("TICK")

        
        ## Final projective measurement layer
        circuit_final_meas = stim.Circuit()
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
            
        circuit = circuit_init + circuit_stab_meas_init + (self.num_cycles - 1)*circuit_stab_meas + circuit_final_meas
        
        # Add noise
        noisy_circuit = circuit
        ## Add CX gate errors
        noisy_circuit = AddCXError(noisy_circuit, 'DEPOLARIZE2(%f)' % self.error_params["p_CX"])

        ## Add measurement errors
        noisy_circuit = AddMeasurementError(noisy_circuit, meas_p=self.error_params["p_m"])
        
        ## Add reset errors
        noisy_circuit = AddResetError(noisy_circuit, reset_p=self.error_params["p_state_p"])
        
        ## Add idling errors
        noisy_circuit = AddIdlingError(noisy_circuit, error_instruction='Z_ERROR(%f)' % (self.error_params["p_i"]),
                                      target_qubit_indices=data_indices)

        self.circuit = noisy_circuit

    
    def _decoding_samples(self, samples):
        '''Decoding the samples obtained from the circuit_sampler'''
        syndrome_historys = [1.0*np.reshape(np.array(sample[:-len(self.eval_code.lx)]), [self.num_cycles + 1, np.shape(self.eval_code.hx)[0]]) for sample in samples]
        logical_values = [1.0*np.array(sample[-len(self.eval_code.lx):]) for sample in samples]
        
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
        eval_func = lambda physical_error_rate: self._single_run()        
        sim_if_error_list = parmap(eval_func, [0]*num_samples, nprocs = mp.cpu_count())
        
#         error_count = np.sum(sim_if_error_list)/num_rounds
        error_count = np.sum(sim_if_error_list)
        # compute logical error rate
        logical_error_rate = error_count/num_samples
        
        logical_error_rate = 1.0 - (1-logical_error_rate)**(1/num_rounds)
        
        logical_error_rate_eb = np.sqrt(
            (1-logical_error_rate)*logical_error_rate/num_samples)

        # compute word error rate
        word_error_rate = 1.0 - \
            (1-logical_error_rate)**(1/self.K)
        word_error_rate_eb = logical_error_rate_eb * \
            ((1-logical_error_rate_eb)**(1/self.K - 1))/self.K
        
        return word_error_rate, word_error_rate_eb