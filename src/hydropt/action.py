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
    

class PowerAction(BaseAction):
    def constrain_power(self, constraints, power):
        
        if constraints is not None and self.turbine in constraints:
            constraint = constraints[self.turbine]
            constrained_power = constraint.transform(power)
        else:    
            constrained_power = power
            
        return constrained_power


class FixedPowerAction(PowerAction):
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

    
class Standing(FixedPowerAction):
    def __init__(self, turbine=None):
        super().__init__(0.0, turbine)


class MinPower(PowerAction):
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
    

class MaxPower(PowerAction):
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


class PowerPlantAction(list):
    def __init__(self, power_plant=None, *args):
        list.__init__(self, *args)
        self.power_plant = power_plant
        
    def turbine_power(self, constraints=None):
        return [action.turbine_power(constraints) for action in self]
    
    def basin_flow_rates(self, constraints=None):
        basins = self.power_plant.basins
        basin_flow_rates = len(basins)*[0]
        
        get_basin_index = self.power_plant.basin_index # method to get index
        
        for action in self:
            outflow_ind = get_basin_index(action.turbine.upper_basin)
            if outflow_ind is not None:
                basin_flow_rates[outflow_ind] += action.flow_rates(constraints)
            inflow_ind = get_basin_index(action.turbine.lower_basin)
            if inflow_ind is not None:
                basin_flow_rates[inflow_ind] -= action.flow_rates(constraints)
                
        return basin_flow_rates
    
    def append(self, action):
        if isinstance(action, BaseAction):
            list.append(self, action)
        else:
            raise ValueError('Object is no instance of BaseAction.')
          
    def __repr__(self):
        action_list = [a for a in self]
        return f"{self.__class__.__name__}({action_list})"
    
    
class PowerPlantActions(list):
    def __init__(self, *args):
        list.__init__(self, *args)
        
    def turbine_power(self, constraints=None):
        return [power_plant_action.turbine_power(constraints) 
                for power_plant_action in self]
        
    def basin_flow_rates(self, constraints=None):
        return [power_plant_action.basin_flow_rates(constraints) 
                for power_plant_action in self]
    
    def __repr__(self):
        action_list = [a for a in self]
        return f"{self.__class__.__name__}({action_list})"
        