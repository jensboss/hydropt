#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 20:47:27 2020

@author: Jens
"""

import numpy as np

            
class BaseAction():
    def __init__(self, turbine=None):
        self.turbine = turbine
        
    def turbine_power(self, constraints=None):
        raise NotImplementedError
        
    def flow_rates(self, constraints=None):
        raise NotImplementedError
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine})"
    

class ActionPower(BaseAction):
    def constrain_power(self, constraints, power):
        
        if constraints is not None and self.turbine in constraints:
            constraint = constraints[self.turbine]
            constrained_power = constraint.transform(power)
        else:    
            constrained_power = power
            
        return constrained_power


class ActionPowerFixed(ActionPower):
    def __init__(self, power, turbine=None):
        super().__init__(turbine)
        self.power = power
        
    def turbine_power(self, constraints=None):
        
        power = self.constrain_power(constraints, self.power)
        
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        
        return power*np.ones((num_plant_states,))
    
    def flow_rates(self, constraints=None):
        return self.turbine.flow_rate(self.power)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine}, power={self.power})"

    
class ActionStanding(ActionPowerFixed):
    def __init__(self, turbine=None):
        super().__init__(0.0, turbine)


class ActionPowerMin(ActionPower):
    def __init__(self, turbine=None):
        super().__init__(turbine)
        
    def turbine_power(self, constraints=None):
        
        power_min = self.constrain_power(constraints, 
                                               self.turbine.base_load)
            
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return power_min*np.ones((num_plant_states,))
    
    def flow_rates(self, constraints=None):
        
        power_min = self.constrain_power(constraints, 
                                               self.turbine.base_load)
        
        return self.turbine.flow_rate(power_min)
    

class ActionPowerMax(ActionPower):
    def __init__(self, turbine=None):
        super().__init__(turbine)
        
    def turbine_power(self, constraints=None):
        
        power_max = self.constrain_power(constraints, 
                                               self.turbine.max_power)
            
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return power_max*np.ones((num_plant_states,))
    
    def flow_rates(self, constraints=None):
        
        power_max = self.constrain_power(constraints, 
                                               self.turbine.max_power)
        
        return self.turbine.flow_rate(power_max)




class ActionFlowRateFixed(BaseAction):
    def __init__(self, flow_rate, turbine=None):
        super().__init__(turbine)
        self.flow_rate = flow_rate
        
    def turbine_power(self, constraints=None):
        return self.turbine.power(self.flow_rate)
    
    def flow_rates(self, constraints=None):
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return self.flow_rate*np.ones((num_plant_states,))
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine}, flow_rate={self.flow_rate})"



class PowerPlantAction():
    def __init__(self, actions, power_plant=None):
        self.actions = tuple(actions)
        self.power_plant = power_plant
        
    def turbine_power(self, constraints=None):
        return [action.turbine_power(constraints) for action in self.actions]
    
    def basin_flow_rates(self, constraints=None):
        basins = self.power_plant.basins
        basin_flow_rates = len(basins)*[0]
        
        get_basin_index = self.power_plant.basin_index # methode to get index
        
        for action in self.actions:
            outflow_ind = get_basin_index(action.turbine.upper_basin)
            if outflow_ind is not None:
                basin_flow_rates[outflow_ind] += action.flow_rates(constraints)
            inflow_ind = get_basin_index(action.turbine.lower_basin)
            if inflow_ind is not None:
                basin_flow_rates[inflow_ind] -= action.flow_rates(constraints)
                
        return basin_flow_rates
        
    def __len__(self):
        return len(self.actions)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.actions})"
    
    
class PowerPlantActions():
    def __init__(self, power_plant_actions):
        self.power_plant_actions = power_plant_actions
        
    def turbine_power(self, constraints=None):
        return [power_plant_action.turbine_power(constraints) 
                for power_plant_action in self.power_plant_actions]
        
    def basin_flow_rates(self, constraints=None):
        return [power_plant_action.basin_flow_rates(constraints) 
                for power_plant_action in self.power_plant_actions]
    
    def __getitem__(self, index):
        return self.power_plant_actions[index]
    
    def __iter__(self):
        return iter(self.power_plant_actions)
    
    def __len__(self):
        return len(self.power_plant_actions)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.power_plant_actions})"
        