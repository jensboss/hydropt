import numpy as np
import time
import pandas as pd

from hydropt.core import backward_induction, forward_propagation, CoreAction
from hydropt.constraints import ConstraintsSeries


class Underlyings():
    def __init__(self, time, price_curve=None, inflow_rate=None):
        self.time = time
        self.price_curve = price_curve
        self.inflow_rate = inflow_rate
        
    def n_steps(self):
        return self.time.shape[0]
    
    def dt(self):
        return (self.time[1]-self.time[0]) / np.timedelta64(1, 's')
    
    

def compute_core_action_series(power_plant, constraints_series, dt):
    unique_core_actions = {}
    core_action_series = []
    
    for contraints in constraints_series.normalized(power_plant.turbines):
        
        key = tuple([tuple(constraint) for constraint in contraints.values()])
        
        if key not in unique_core_actions:
            
            core_actions = []
            for pp_action in power_plant.actions():
                core_actions.append(
                    CoreAction(
                        np.array(pp_action.turbine_power(contraints)), 
                        np.array(pp_action.basin_flow_rates(contraints))*dt, 
                        power_plant.basin_volumes(), 
                        power_plant.basin_num_states())
                    )
            
            unique_core_actions[key] = core_actions      
            
        core_action_series.append(unique_core_actions[key])
        
    return core_action_series
    
        
class Scenario():
    def __init__(self, power_plant, underlyings, constraints=None, 
                 water_value_end=0, basin_limit_penalty=1e14*3600, name=None):
        
        self.power_plant = power_plant
        self.underlyings = underlyings
        
        self.start_time = underlyings.time[0]
        self.end_time = underlyings.time[-1] + (underlyings.time[1] - underlyings.time[0])
        
        if constraints is None:
            self.constraints_series = ConstraintsSeries(self.start_time, self.end_time)
        else:
            self.constraints_series = ConstraintsSeries(self.start_time, 
                                                        self.end_time,
                                                        constraints)
            
        self.water_value_end = water_value_end
        self.name = name
        
        self.basin_limit_penalty = basin_limit_penalty
        

    def run(self):
        n_steps = self.underlyings.n_steps()
        dt = self.underlyings.dt()
        price_curve = self.underlyings.price_curve
        
        inflow = self.underlyings.inflow_rate*dt
        
        volume = self.power_plant.basin_volumes()
        num_states = self.power_plant.basin_num_states()
        basins_init_volumes = self.power_plant.basin_start_volumes()
        
        penalty = self.basin_limit_penalty  
        
        water_value_end = self.water_value_end
        
        t_start = time.time()
        
        # make core actions
        action_series = compute_core_action_series(self.power_plant, self.constraints_series, dt)
        
                
        action_grid, value_grid = backward_induction(
            n_steps, 
            volume, 
            num_states, 
            action_series, 
            inflow, 
            price_curve, 
            water_value_end, 
            penalty)
        
        t_end = time.time()
        print(t_end-t_start)
        
        t_start = time.time()
        turbine_act_taken, basin_act_taken, vol = forward_propagation(
            n_steps, 
            volume, 
            num_states, 
            basins_init_volumes,
            action_series, 
            inflow, 
            action_grid)
        
        t_end = time.time()
        print(t_end-t_start)
        
        self.action_grid_ = action_grid
        self.value_grid_ = value_grid
        
        self.turbine_actions_ = turbine_act_taken
        self.basin_actions_ = basin_act_taken
        self.volume_ = vol
        

        actions_taken = pd.DataFrame(
            index=self.underlyings.time,
            data=turbine_act_taken/1e6,
            columns=[t.name + ' (MWatt)' for t in self.power_plant.turbines]
        )
        
        volumes_seen = pd.DataFrame(
            index=self.underlyings.time,
            data=vol[0:-1, :]/1e6,
            columns=[b.name + ' (Mio. m3)' for b in self.power_plant.basins]
        )
        
        price_curve_frame = pd.DataFrame(
            index=self.underlyings.time,
            data=self.underlyings.price_curve,
            columns=['price curve (EUR/MWh)',]
        )
        
        self.results_ = price_curve_frame.join(actions_taken).join(volumes_seen)
        
        
    def valuation(self):
        if self.turbine_actions_ is None:
            RuntimeError('Need to run scenario first.')
            
        return np.dot(self.turbine_actions_.T, self.underlyings.price_curve).sum()/1e6
        
        

            

