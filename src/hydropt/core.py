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


def simple_trans_matrix(V, num_states, q):
    L_combined = sparse.dia_matrix(np.ones((1,1)))
    for k in np.arange(V.shape[0]):
        # state with index 0 represents empty basin
        L = sparse.diags(*transition_prob(V[k], num_states[k], q[k]),
                         shape=(num_states[k],num_states[k]),
                         dtype=np.float64)
        L_combined = sparse.kron(L_combined, L)
    return L_combined.tocsc()


def kron_index(num_states, position):
    n = num_states[position]
    rep_d1 = np.prod(num_states[0:position], dtype=np.int)
    rep_d2 = np.prod(num_states[position:], dtype=np.int)//n
    
    index_rep = np.tile(np.arange(n), (rep_d2, rep_d1))
    return index_rep.flatten('F')


def kron_indices(num_states, positions):
    indices = list()
    
    for position in positions:
        indices.append(kron_index(num_states, position))
        
    return np.stack(indices, axis=1)


def kron_basis_map(num_states):
    num_states = np.int64(np.round(num_states))
    return np.flip(np.cumprod(np.flip(num_states)))//num_states


class CoreAction():
    def __init__(self, turbine_action, basin_action, volumes, num_states):
        
        self.turbine_action = turbine_action
        self.basin_action = basin_action
        
        self.volumes = volumes
        self.num_states = num_states
        
        self._trans_matrix = None
        
    def trans_matrix(self):
        
        if self._trans_matrix is None:
            self._trans_matrix = trans_matrix(self.volumes, self.num_states, self.basin_action)
        
        return self._trans_matrix
        

def backward_induction(n_steps, volume, num_states, action_series,
                       inflows, prices, water_value_end, penalty):
   
    
    num_states_tot = np.prod(num_states)
    
    # initialize boundary condition (valuation of states at the end of optimization)
    value = np.zeros((num_states_tot, ))
    for k in np.arange(num_states.shape[0]):
        value += water_value_end*volume[k]*np.linspace(0,1, num_states[k])[kron_index(num_states, k)]
    
    # allocate momory
    rewards_to_evaluate = np.zeros((len(action_series[0]), num_states_tot))
    
    action_grid = []
    value_grid = []
    
    
    # loop backwards through time (backward induction)
    for backward_step_index in np.flip(np.arange(n_steps)):
        price = prices[backward_step_index]
        inflow = inflows[backward_step_index, :]
        actions = action_series[backward_step_index]
        
        # compute inflow transition matrix, which changes only with time
        L_inflow = simple_trans_matrix(volume, num_states, -inflow)
        
        for action_index, action in enumerate(actions):
                
            L = L_inflow @ action.trans_matrix()
            
            immediate_reward = np.sum(action.turbine_action*price, axis=0)
            future_reward = L.T.dot(value) 
            
            # TODO: Normalize penalty
            penatly_reward = penalty*(1-np.sum(L, axis=0))
            
            rewards_to_evaluate[action_index, :] = future_reward + immediate_reward - penatly_reward

        # find index of optimal action for each state
        optimal_action_index = np.argmax(rewards_to_evaluate, axis=0)
        
        # value of each state is given by the reward of the optimal action
        value = rewards_to_evaluate[optimal_action_index, np.arange(num_states_tot)]
                
        action_grid.append(optimal_action_index)
        value_grid.append(value)
    
    return np.flipud(np.array(action_grid)), np.flipud(np.array(value_grid))


def forward_propagation(n_steps, volume, num_states, basins_contents, action_series,
                        inflow, action_grid):
    # TODO: Clean up.
    basin_actions_taken = np.zeros((n_steps, action_series[0][0].basin_action.shape[0]))
    turbine_actions_taken = np.zeros((n_steps, action_series[0][0].turbine_action.shape[0]))
    
    vol = np.zeros((n_steps+1, volume.shape[0]))
    vol[0,:] = basins_contents
    state_finder = kron_basis_map(num_states)
    
    for step_index, actions in enumerate(action_series):
        
        basin_actions = np.array([action.basin_action for action in actions])
        turbine_actions = np.array([action.turbine_action for action in actions])
        
        state_index = np.dot(state_finder, np.int64(np.round((num_states-1)*vol[step_index, :]/volume)))
        
        try:
            basin_actions_taken[step_index] = basin_actions[action_grid[step_index, state_index]][:,state_index]
        except IndexError as e:
            print(e)
            
        try:     
            turbine_actions_taken[step_index] = turbine_actions[action_grid[step_index, state_index]][:,state_index]
        except IndexError as e:
            print(e)
            
        vol[step_index+1, :] = vol[step_index,:] - basin_actions_taken[step_index, :] + inflow[step_index, :]
        
    return turbine_actions_taken, basin_actions_taken, vol


    
def transition_coo_matrix_params(vol, num_states, q, basin_index):
 
    # number of product states
    m = np.prod(num_states)
    # range of product space indices
    j = np.arange(m)
    # volume defference between states for this basin
    dvols = vol/(num_states[basin_index]-1)
    # index step between different indices for this basin in kron matrix
    basis_map = kron_basis_map(num_states)[basin_index]
    # state indices for this basin in kron space
    index = kron_index(num_states, basin_index)
    # compute state indices change (interpolate -> floor/ceil)
    dk_floor = np.int64(q/dvols)
    dk_ceil = np.int64(dk_floor + np.sign(q))
    # compute target state indices (basin state index)
    k_floor = index - dk_floor
    k_ceil =  index - dk_ceil
    # make sure new indices are not out of bound
    valid_floor = (k_floor < num_states[basin_index]) & (k_floor >= 0)
    valid_ceil = (k_ceil < num_states[basin_index]) & (k_ceil >= 0)
    # compute target product state indices
    i_floor = j-dk_floor*basis_map
    i_ceil = j-dk_ceil*basis_map
    # compute weights
    p_ceil = (np.abs(q) % dvols)/dvols
    p_floor = 1 - p_ceil
    
    # ! duplicates are ok because their get summed up!
    
    # prepare return
    data = np.concatenate((p_floor[valid_floor],
                           p_ceil[valid_ceil]))
    
    i_comb = np.concatenate((i_floor[valid_floor],
                             i_ceil[valid_ceil]))
    
    j_comb = np.concatenate((j[valid_floor],
                             j[valid_ceil]))
    
    return data, (i_comb, j_comb), (m,m)
    

def trans_matrix(vols, num_states, q):
    L_combined = None
    for k in np.arange(vols.shape[0]):
        data, coords, shape = transition_coo_matrix_params(vols[k], num_states, q[k], k)
        L = sparse.coo_matrix((data, coords), shape=shape).tocsc()
        if L_combined is None:
            L_combined = L
        else:
            L_combined = L_combined @ L
    return L_combined




