#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:14:02 2020

@author: Jens
"""

import numpy as np
import scipy.sparse as sparse

def transition_prob(V, num_states, q):
    if q == 0:
        return ([1.0, ], [0, ])
    else:
        dV = V/(num_states-1)
        k_0 = int(q/dV)
        p_0 = 1 - (np.abs(q) % dV)/dV
        k_1 = int(k_0 + np.sign(q))
        p_1 = (np.abs(q) % dV)/dV
        return ([p_0, p_1], [k_0, k_1])


def transition_matrix(V, num_states, q):
    L_combined = sparse.dia_matrix(np.ones((1,1)))
    for k in np.arange(V.shape[0]):
        # state with index 0 represents empty basin
        L = sparse.diags(*transition_prob(V[k], num_states[k], q[k]),
                         shape=(num_states[k],num_states[k]),
                         dtype=np.float64)
        L_combined = sparse.kron(L_combined, L)
    return L_combined



def kron_index(num_states, position):
    index = np.ones(1, dtype=np.int64)
    
    for k, num in enumerate(num_states):
        if k == position:
            index = np.kron(index, np.arange(num))
        else:
            index = np.kron(index, np.ones(num, dtype=np.int64))
            
    return index


def kron_indices(num_states, positions):
    indices = list()
    
    for position in positions:
        indices.append(kron_index(num_states, position))
        
    return np.stack(indices, axis=1)


def kron_basis_map(num_states):
    num_states = np.int64(np.round(num_states))
    return np.flip(np.cumprod(np.flip(num_states)))//num_states

def kron_action(q, m):
    return [f*np.ones((m,)) for f in q]


def backward_induction(n_steps, volume, num_states, turbine_actions, basin_actions,
                       inflow, hpfc, water_value_end, penalty):
    num_states_tot = np.prod(num_states)
    
    # initialize boundary condition (valuation of states at the end of optimization)
    value = np.zeros((num_states_tot, ))
    for k in np.arange(num_states.shape[0]):
        value += water_value_end*volume[k]*np.linspace(0,1, num_states[k])[kron_index(num_states, k)]
    
    # allocate momory
    rewards_to_evaluate = np.zeros((turbine_actions.shape[0], num_states_tot))
    action_grid = np.zeros((n_steps, num_states_tot), dtype=np.int64)
    value_grid = np.zeros((n_steps,num_states_tot))
    
    # loop backwords through time (backward induction)
    for k in np.arange(n_steps):
        hpfc_now = hpfc[(n_steps-1)-k]
        inflow_now = inflow[(n_steps-1)-k, :]
        
        # loop through all actions and every state
        for act_index, (turbine_action, basin_action) in enumerate(zip(turbine_actions, basin_actions)):
            # L = transition_matrix(volume, num_states, basin_action-inflow_now)
            L = trans_matrix_2(volume, num_states, kron_action(basin_action-inflow_now, num_states_tot))
            # if not np.all(L.todense()==L2.todense()):
            #     print(L.todense(), L2.todense())
            immediate_reward = np.sum(turbine_action*hpfc_now)
            future_reward = L.T.dot(value) 
            # TODO: Normalize penalty
            penatly_reward = penalty*(1-np.sum(L, axis=0))
            rewards_to_evaluate[act_index, :] = future_reward + immediate_reward - penatly_reward

        # find index of optimal action for each state
        optimal_action_index = np.argmax(rewards_to_evaluate, axis=0)
        
        # value of each state is given by the reward of the optimal action
        value = rewards_to_evaluate[optimal_action_index, np.arange(num_states_tot)]
        
        # fill action and value grids
        action_grid[(n_steps-1)-k, :] = optimal_action_index
        value_grid[(n_steps-1)-k, :] = value
    
    return action_grid, value_grid


def forward_propagation(n_steps, volume, num_states, basins_contents, turbine_actions, basin_actions,
                        inflow, action_grid):
    basin_actions_taken = np.zeros((n_steps, basin_actions.shape[1]))
    turbine_actions_taken = np.zeros((n_steps, turbine_actions.shape[1]))
    
    vol = np.zeros((n_steps+1, volume.shape[0]))
    vol[0,:] = basins_contents
    state_finder = kron_basis_map(num_states)
    
    for k in np.arange(n_steps):
        state_index = np.dot(state_finder, np.int64(np.round((num_states-1)*vol[k, :]/volume)))
        basin_actions_taken[k] = basin_actions[action_grid[k, state_index]]
        turbine_actions_taken[k] = turbine_actions[action_grid[k, state_index]][:,state_index]
        vol[k+1, :] = vol[k,:] - basin_actions_taken[k, :] + inflow[k, :]
        
    return turbine_actions_taken, basin_actions_taken, vol


def transition_coo_matrix_params(vols, num_states, q, basin_index):

    def valid_index_mask(i, num_states):
        not_too_large = i < num_states
        not_too_small = i >= 0
        return not_too_large & not_too_small
        
    m = np.prod(num_states)
    j = np.arange(m)
    
    dvols = vols/(num_states[basin_index]-1)
    
    basis_map = kron_basis_map(num_states)[basin_index]
    
    index = kron_index(num_states, basin_index)
    
    dk_floor = np.int64(q/dvols)
    dk_ceil = np.int64(dk_floor + np.sign(q))
    k_floor = index + dk_floor
    k_ceil =  index + dk_ceil
    
    valid_floor = valid_index_mask(k_floor, num_states[basin_index])
    valid_ceil = valid_index_mask(k_ceil, num_states[basin_index])
    
    i_floor = j+dk_floor*basis_map
    i_ceil = j+dk_ceil*basis_map
    
    p_ceil = (np.abs(q) % dvols)/dvols
    p_floor = 1 - p_ceil
    
    data = np.concatenate((p_floor[valid_floor],
                           p_ceil[valid_ceil]))
    
    i_comb = np.concatenate((i_floor[valid_floor],
                             i_ceil[valid_ceil]))
    
    j_comb = np.concatenate((j[valid_floor],
                             j[valid_ceil]))
    
    return data, (i_comb, j_comb), (m,m)
    

def trans_matrix_2(vols, num_states, q):
    L_combined = None
    for k in np.arange(vols.shape[0]):
        data, coords, shape = transition_coo_matrix_params(vols[k], num_states, q[k], k)
        L = sparse.coo_matrix((data, coords), shape=shape).todia()
        if L_combined is None:
            L_combined = L
        else:
            L_combined = L_combined @ L
    return L_combined.T



if __name__ == '__main__':
    from scipy.sparse import coo_matrix
        
    
    def transition_coo_matrix_entries(vols, num_states, q):
    
        def valid_index_mask(i, num_states):
            not_too_large = i < num_states
            not_too_small = i >= 0
            return not_too_large & not_too_small
            
            
        dvols = vols/(num_states-1)
        
        basis_map = kron_basis_map(num_states)
        
        index = np.column_stack((kron_index(num_states, 0), 
                                 kron_index(num_states, 1)))
        
        k_floor = index+np.int64(np.floor(q/dvols))
        k_ceil =  index+np.int64(np.ceil(q/dvols))
        
        valid_floor = valid_index_mask(k_floor, num_states)
        valid_ceil = valid_index_mask(k_ceil, num_states)
        
        mask_floor_floor = np.all(valid_floor, axis=1)
        mask_ceil_ceil = np.all(valid_ceil, axis=1)
        mask_ceil_floor = valid_ceil[:,0] & valid_floor[:,1]
        mask_floor_ceil = valid_ceil[:,1] & valid_floor[:,0]
        
        i_floor_floor = k_floor.dot(basis_map)
        i_ceil_ceil = k_ceil.dot(basis_map)
        i_ceil_floor = np.column_stack((k_ceil[:,0],k_floor[:,1])).dot(basis_map)
        i_floor_ceil = np.column_stack((k_floor[:,0],k_ceil[:,1])).dot(basis_map)
        
        p_ceil_temp = (np.abs(q) % dvols)/dvols
        p_floor_temp = 1 - p_ceil_temp
        
        p_ceil_ceil = np.prod(p_ceil_temp, axis=1)
        p_floor_floor = np.prod(p_floor_temp, axis=1)
        
        p_ceil_floor = p_ceil_temp[:,0]*p_floor_temp[:,1]
        p_floor_ceil = p_floor_temp[:,0]*p_ceil_temp[:,1]
        
        m = np.prod(num_states)
        j = np.arange(m)
        
        data = np.concatenate((p_ceil_ceil[mask_ceil_ceil],
                               p_floor_floor[mask_floor_floor],
                               p_ceil_floor[mask_ceil_floor],
                               p_floor_ceil[mask_floor_ceil]))
        
        i_comb = np.concatenate((i_ceil_ceil[mask_ceil_ceil],
                                 i_floor_floor[mask_floor_floor],
                                 i_ceil_floor[mask_ceil_floor],
                                 i_floor_ceil[mask_floor_ceil]))
        
        j_comb = np.concatenate((j[mask_ceil_ceil],
                                 j[mask_floor_floor],
                                 j[mask_ceil_floor],
                                 j[mask_floor_ceil]))
        
        return data, (i_comb, j_comb), (m,m)
        
        
    def trans_matrix(vols, num_states, q):
        data, coords, shape = transition_coo_matrix_entries(vols, num_states, q)
        matrix = coo_matrix((data, coords), shape=shape).todia()
        return matrix.T
    

    
    vols = np.array([10.0, 7.0])
    num_states = np.array([3, 2])
    

    
    # q = np.column_stack((-np.array([0.0,5.0,6.0]), np.array([0.0,5.0,6.0])))
    q = np.array([5.0, -5.0])
    q_ = np.column_stack((np.array([5.0, 5.0, 5.0]), -np.array([5.0, 5.0, 5.0])))
    print(q)
    qq = q_[kron_index(num_states, 0),:]
    print(qq)
    
    qqq = qq.T
    
    # print(trans_matrix(vols, num_states, qq).todense())
    
    print(transition_matrix(vols, num_states, q).todense())
    
    print(trans_matrix_2(vols, num_states, qqq).todense())
    
    # print(p_floor, p_ceil)
    # print((index+k_floor).dot(basis_map), (index+k_ceil).dot(basis_map))
    
    # ret =  ([p_0, p_1], [k_0, k_1])



