#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:14:02 2020

@author: Jens
"""

import numpy as np

from dynprog.core import kron_index, kron_indices



class BasinLevels():
    def __init__(self, empty, full=None, basin=None, vol_to_level_lut=None):
        self.basin = basin
        self.empty = empty
        
        if full is None:
            self.full = empty
        else:
            self.full = full
            
        self.vol_to_level_lut = vol_to_level_lut
        
        self._values = None
        
    @property
    def values(self):
        if self._values is None and self.vol_to_level_lut is None:
            self._values = np.linspace(self.empty, self.full, self.basin.num_states)
        return self._values


class Basin():
    def __init__(self, volume, num_states, init_volume, levels, name=None, power_plant=None):
        self.name = name
        self.volume = volume
        self.num_states = num_states
        self.init_volume = init_volume
        
        self._levels = None
        self.levels = levels
        
        self.power_plant = power_plant
        
    def index(self):
        if self.power_plant is None:
            return None
        else:
            return self.power_plant.basin_index(self)
       
    @property
    def levels(self):
        return self._levels
        
    @levels.setter
    def levels(self, levels):
        if isinstance(levels, BasinLevels):
            self._levels = levels
            self._levels.basin = self
        elif isinstance(levels, tuple):
            self._levels = BasinLevels(empty=levels[0], full=levels[1], basin=self)
        else:
            self._levels = BasinLevels(levels, basin=self)
            
    def kron_levels(self):
        if self.power_plant is None:
            return self._levels.empty
        else:
            basin_num_states = self.power_plant.basin_num_states()
            index = self.power_plant.basin_index(self)
            return self._levels.values[kron_index(basin_num_states, index)]
        
    def __repr__(self):
        if self.name is None:
            name = f'basin_{self.index()}'
        return f"Basin('{name}', {self.volume}, {self.num_states})"

    
class Outflow(Basin):
    def __init__(self, outflow_level, name='Outflow'):
        super().__init__(volume=1, num_states=2, init_volume=0, levels=outflow_level, name=name)
        


class BaseAction():
    def __init__(self, turbine=None):
        self.turbine = turbine
        
    def turbine_power(self):
        raise NotImplementedError
        
    def basin_flow_rates(self):
        raise NotImplementedError
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine})"


class ActionPowerFixed(BaseAction):
    def __init__(self, power, turbine=None):
        super().__init__(turbine)
        self.power = power
        
    def turbine_power(self):
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return self.power*np.ones((num_plant_states,))
    
    def basin_flow_rates(self):
        return self.turbine.flow_rate(self.power)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine}, power={self.power})"

    
class ActionFlowRateFixed(BaseAction):
    def __init__(self, flow_rate, turbine=None):
        super().__init__(turbine)
        self.flow_rate = flow_rate
        
    def turbine_power(self):
        return self.turbine.power(self.flow_rate)
    
    def basin_flow_rates(self):
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return self.flow_rate*np.ones((num_plant_states,))
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.turbine}, flow_rate={self.flow_rate})"

    
class ActionStanding(ActionFlowRateFixed):
    def __init__(self, turbine=None):
        super().__init__(0.0, turbine)
        
        
class ActionMin(BaseAction):
    def __init__(self, turbine=None):
        super().__init__(turbine)
        
    def turbine_power(self):
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return self.turbine.base_load*np.ones((num_plant_states,))
    
    def basin_flow_rates(self):
        return self.turbine.flow_rate(self.turbine.base_load)

        
class ActionMax(BaseAction):
    def __init__(self, turbine=None):
        super().__init__(turbine)
        
    def turbine_power(self):
        num_plant_states = self.turbine.upper_basin.power_plant.num_states()
        return self.turbine.max_power*np.ones((num_plant_states,))
    
    def basin_flow_rates(self):
        return self.turbine.flow_rate(self.turbine.max_power)
    
    
    
class PlantAction():
    def __init__(self, actions, power_plant=None):
        self.actions = tuple(actions)
        self.power_plant = power_plant
        
    def turbine_power(self):
        return [action.turbine_power() for action in self.actions]
    
    def basin_flow_rates(self):
        basins = self.power_plant.basins
        basin_flow_rates = len(basins)*[0]
        for action in self.actions:
            outflow_ind = action.turbine.upper_basin.index()
            if outflow_ind is not None:
                basin_flow_rates[outflow_ind] += action.basin_flow_rates()
            inflow_ind = action.turbine.lower_basin.index()
            if inflow_ind is not None:
                basin_flow_rates[inflow_ind] -= action.basin_flow_rates()
                
        return basin_flow_rates
        
    def __repr__(self):
        return f"{self.__class__.__name__}({self.actions})"
    
    
class ActionCollection():
    def __init__(self, product_actions):
        self.product_actions = product_actions
        
    def turbine_power(self):
        return [product_action.turbine_power() for product_action in self.product_actions]
    
    def basin_flow_rates(self):
        return [product_action.basin_flow_rates() for product_action in self.product_actions]
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.product_actions})"
        

        
class Turbine():
    def __init__(self, name, max_power, base_load, 
                 upper_basin, lower_basin, efficiency, actions=None):
        self.name = name
        self.max_power = max_power
        self.base_load = base_load
        self.upper_basin = upper_basin
        self.lower_basin = lower_basin
        
        self.efficiency = efficiency
        
        self._actions = []
        if actions is not None:
            self.actions = actions
        
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, actions):
        for action in actions:
            action.turbine = self
        self._actions = actions
    
    def head(self):
        return self.upper_basin.kron_levels() - self.lower_basin.kron_levels()
    
    def power(self, flow_rate):
        return self.efficiency*(1000*9.81)*self.head()*flow_rate
    
    def flow_rate(self, power):
        return (power/(self.efficiency*(1000*9.81)))/self.head()
    
    def __repr__(self):
        return f"Turbine('{self.name}')"


    
class Plant():
    def __init__(self, basins=None, turbines=None, constraints=None):
        self._basins = []
        self._basin_index = {}
        self.add_basins(basins)      
        self.turbines = turbines
        self.constraints = constraints
        
    @property
    def basins(self):
        return self._basins
    
    def add_basins(self, basins):
        for basin in basins:
            basin.power_plant = self
            self._basins.append(basin)
            self._basin_index[basin] = len(self._basins)-1
        
    def basin_index(self, basin):
        return self._basin_index[basin]
        
    def basin_volumes(self):
        return np.array([basin.volume for basin in self.basins])
    
    def basin_num_states(self):
        return np.array([basin.num_states for basin in self.basins])
    
    def basin_init_volumes(self):
        return np.array([basin.init_volume for basin in self.basins])
    
    def basin_names(self):
        return np.array([basin.names for basin in self.basins])
    
    def num_states(self):
        return np.prod(self.basin_num_states())
    
    def turbine_actions(self):
        actions = list()
        for turbine in self.turbines:
            actions.append(turbine.actions)
        return actions
        
    def actions(self):
        turbine_actions = self.turbine_actions()
        num_actions = [len(a) for a in turbine_actions]
        combinations = kron_indices(num_actions, range(len(num_actions)))
        product_actions = []
        for comb in combinations:
            group = []
            for k in range(len(comb)):
                group.append(turbine_actions[k][comb[k]])
            product_actions.append(PlantAction(group, self))
        return ActionCollection(product_actions)
    
            

class Underlyings():
    def __init__(self, price_curve, inflows):
        self.price_curve = price_curve
        self.inflows = inflows    