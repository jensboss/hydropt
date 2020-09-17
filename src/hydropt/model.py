import numpy as np

from hydropt.core import kron_index, kron_indices
from hydropt.action import PowerPlantActions, PowerPlantAction


class BasinLevels():
    """The basin levels correspond to the fill-levels of a basin given its 
    linearly spaced volume states.
    """
    
    def __init__(self, empty, full=None, basin=None, vol_to_level_lut=None, 
                 basin_shape='wedge'):
        
        self.basin = basin
        self.empty = empty
        
        if full is None:
            self.full = empty
        else:
            self.full = full
            
        if vol_to_level_lut is None:
            self.vol_to_level_lut = self.compute_vol_to_level_lut(basin_shape)
        else:
            self.vol_to_level_lut = vol_to_level_lut
        
        
    def compute_vol_to_level_lut(self, basin_shape='wedge'):
        
        vols = np.linspace(0, self.basin.volume, self.basin.num_states)
        
        if basin_shape == 'wedge':
            
            if self.empty < self.full:
                
                height = self.full - self.empty
                A = height**2 # width = 2*height
                length = self.basin.volume/A
                
                levels = np.sqrt(vols/length)+self.empty
                
            else:
                levels = self.empty*np.ones(self.basin.num_states)
                
            
        else:
            raise ValueError(f"Basin shape '{basin_shape}' not implemented. "
                             "Choose from the following models ['wegde', ]")
            
        return np.stack((vols, levels)).T
        
    @property
    def values(self):
        
        if self.basin is None:
            
            return np.array()
            
        else:
            
            vols = np.linspace(0, self.basin.volume, self.basin.num_states)
            
            vols_p = self.vol_to_level_lut[:,0]
            levels_p = self.vol_to_level_lut[:,1]
            
            return np.interp(vols, vols_p, levels_p)

        
    def __repr__(self):
        return f"BasinLevels(basin='{self.basin}', {self.values})"


class Basin():
    """The basin is a reservoir of water. Each basin specifies a discrete set 
    of states. The state space of the power plant is defined by the tensor
    product between the states of the individual basins.
    """
    
    def __init__(self, name, volume, num_states, levels, 
                 start_volume, end_volume=0, power_plant=None):
        self.name = name
        self.volume = volume
        self.num_states = num_states
                
        self.levels = levels
        
        self.start_volume = start_volume
        self.end_volume = end_volume
        
        self.power_plant = power_plant
        
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
        return f"Basin('{self.name}')"

    
class Outflow(Basin):
    def __init__(self, outflow_level, name='Outflow'):
        super().__init__(volume=1, num_states=2, levels=outflow_level,
                         start_volume=0, name=name)
        

 
class Turbine():
    """The turbine characterices the amount of power that is generated when 
    water flows from the upper to the lower basin.
    """
    
    def __init__(self, name, max_power, base_load, 
                 upper_basin, lower_basin, efficiency, actions=None):
        self.name = name
        self.max_power = max_power
        self.base_load = base_load
        self.upper_basin = upper_basin
        self.lower_basin = lower_basin
        
        self.efficiency = efficiency
        
        self.actions = actions
    
    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, actions):
        self._actions = []
        for action in actions:
            action.turbine = self
            self._actions.append(action)
    
    def head(self):
        return self.upper_basin.kron_levels() - self.lower_basin.kron_levels()
    
    def power(self, flow_rate):
        return self.efficiency*(1000*9.81)*self.head()*flow_rate
    
    def flow_rate(self, power):
        return (power/(self.efficiency*(1000*9.81)))/self.head()
    
    def __repr__(self):
        return f"Turbine('{self.name}')"
   

    
class PowerPlant():
    def __init__(self, basins=None, turbines=None, constraints=None, name=''):
        self.basins = basins    
        self.turbines = turbines
        self.constraints = constraints
        self.name = name
        
    @property
    def basins(self):
        return self._basins
    
    @basins.setter
    def basins(self, basins):
        for basin in basins:
            basin.power_plant = self
        self._basins = basins
            
    def basin_index(self, basin):
        try:
            return self.basins.index(basin)
        except ValueError:
            return None # intended for outflow basin
        
    def basin_volumes(self):
        return np.array([basin.volume for basin in self.basins])
    
    def basin_num_states(self):
        return np.array([basin.num_states for basin in self.basins])
    
    def basin_start_volumes(self):
        return np.array([basin.start_volume for basin in self.basins])
    
    def basin_names(self):
        return [basin.names for basin in self.basins]
    
    def num_states(self):
        return np.prod(self.basin_num_states())
    
    def turbine_actions(self):
        return [turbine.actions for turbine in self.turbines]
        
    def actions(self):
        turbine_actions = self.turbine_actions()
        num_actions = [len(a) for a in turbine_actions]
        combinations = kron_indices(num_actions, range(len(num_actions)))
        power_plant_actions = PowerPlantActions()
        for comb in combinations:
            pp_action = PowerPlantAction(self)
            for k in range(len(comb)):
                pp_action.append(turbine_actions[k][comb[k]])
            power_plant_actions.append(pp_action)
        return power_plant_actions
    
    def summary(self):
        print("--------------------------------------------------")
        print(f"Summary for Power Plant {self.name}:")
        print("--------------------------------------------------")
        print(f"Basins ({len(self.basins)}):")
        for basin in self.basins:
            print(f" - {basin.name}")
            print(f"         volume:   {basin.volume}m3")
            print(f"       # states:   {basin.num_states}")
        print("--------------------------------------------------")
        print(f"Turbines ({len(self.turbines)}):")
        for turbine in self.turbines:
            print(f" - {turbine.name}: {turbine.upper_basin.name} --> {turbine.lower_basin.name}")
            print(f"     max. power:   {turbine.max_power/1e6}MW")
            print(f"       baseload:   {turbine.base_load/1e6}MW")
            print(f"     efficiency:   {turbine.efficiency*100}%")
        print("--------------------------------------------------")
    
            

        
        
