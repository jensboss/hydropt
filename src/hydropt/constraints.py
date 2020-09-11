import functools

import numpy as np


def forgive_none(func):
    
    @functools.wraps(func)
    def wrapper_decorator(a, b):
        
        if a is not None and b is not None:
            return func(a, b)
        elif a is None and b is not None:
            return b
        elif a is not None and b is None:
            return a
        else:
            return None
        
    return wrapper_decorator

@forgive_none
def minimum(a, b):
    return np.minimum(a, b)

@forgive_none
def maximum(a, b):
    return np.maximum(a, b)


class TurbineConstraint():
    def __init__(self, turbine, time_start, time_end, name='', 
                 power_max=np.inf, power_min=-np.inf, 
                 margin_max=-0, margin_min=0):
        
        self.turbine = turbine
        self.time_start = np.datetime64(time_start)
        self.time_end = np.datetime64(time_end)
        
        self.name = name
        
        self.power_max = power_max
        self.power_min = power_min
        self.margin_max = margin_max
        self.margin_min = margin_min
        
        self.validate()
        
    def upper_bound(self):
        return minimum(self.power_max, self.turbine.max_power) + self.margin_max
    
    def lower_bound(self):
        
        lower_bound_temp = maximum(self.power_min, 0) + self.margin_min
        
        if lower_bound_temp > 0:
            return maximum(self.power_min, self.turbine.base_load) + self.margin_min
        else:
            return lower_bound_temp 
        
    def validate(self):
        
        if self.upper_bound() < self.lower_bound():
            raise ValueError("Constraint ill defined.")
        
    def update(self, power_max=None, power_min=None,
               margin_max=None, margin_min=None):
        
        if power_max is not None:
            self.power_max = power_max
            
        if power_min is not None:
            self.power_min = maximum(self.power_min, power_min)
        
        if margin_max is not None:
            self.margin_max = minimum(self.margin_max, margin_max)
            
        if margin_min is not None:
            self.margin_min = maximum(self.margin_min, margin_min)
        
        self.validate()
        
    def transform(self, power):
        return minimum(maximum(power, self.lower_bound()), self.upper_bound())
        
    def __add__(self, other):
        if self.turbine is not other.turbine:
            raise ValueError("Cannot add constraints referring to different turbines. "
                             f"{self.turbine}, {other.turbine}")
        
        name = '+'.join((self.name,other.name))
            
        time_start = maximum(self.time_start, other.time_start)
        time_end = minimum(self.time_end, other.time_end)
        
        return TurbineConstraint(
            turbine=self.turbine,
            time_start=time_start,
            time_end=time_end,
            name=name,
            power_max=minimum(self.power_max, other.power_max),
            power_min=maximum(self.power_min, other.power_min),
            margin_max=minimum(self.margin_max, other.margin_max),
            margin_min=maximum(self.margin_min, other.margin_min)
            )
    
    def __iter__(self):
        for value in (self.turbine, 
                      self.power_max, self.power_min, 
                      self.margin_max, self.margin_min):
            yield value
            
    def __eq__(self, other):
        return tuple(self) == tuple(other)
    
    def __hash__(self):
        return hash(tuple(self))
            
    def __repr__(self):
        return (f"{self.__class__.__name__}({self.turbine},'{self.name}',"
                f"'{self.time_start}','{self.time_end}')")
            
                
class ConstraintsSeries():
    def __init__(self, time_start, time_end, constraints=None):
        
        self._time_start = np.datetime64(time_start)
        self._time_end =  np.datetime64(time_end)
        
        self._time = np.arange(self._time_start, self._time_end)
       
        indices = np.arange(len(self._time))
        
        self._data = [{} for i in indices]
        
        if constraints is None:
            constraints = []
        
        for constraint in constraints:
            ind = indices[(self._time>=constraint.time_start) & (self._time<constraint.time_end)]
            
            for i in ind:
                
                const_dict = self._data[i]
                
                if constraint.turbine in const_dict:
                    const_dict[constraint.turbine] += constraint
                else:
                    const_dict[constraint.turbine] = constraint
                    
    def normalized(self, turbines):
        normalized_data = []
        
        for constraints in self._data:
            
            normalized_constraints = {}
            
            for turbine in turbines:
                
                if turbine in constraints:
                    
                    normalized_constraints[turbine] = constraints[turbine]
                    
                else:
                    
                    normalized_constraints[turbine] = TurbineConstraint(
                        turbine, self.time_start, self.time_end)
                    
            normalized_data.append(normalized_constraints)
                    
        return normalized_data
                    
    @property
    def time(self):
        return self._time
    
    @property
    def data(self):
        return self._data
    
    @property
    def time_start(self):
        return self._time_start
    
    @property
    def time_end(self):
        return self._time_end
    
    def __getitem__(self, index):
        return self._data[index]
    
    def __iter__(self):
        return iter(self._data)
    
    
            
if __name__ == '__main__':
    from dynprog.model import Basin, Outflow, Turbine, PowerPlant
    from dynprog.action import ActionStanding, ActionPowerMin, ActionPowerMax
    
    from dynprog.core import CoreAction


    basins = [Basin(name='basin_1', 
                    volume=81*3600, 
                    num_states=81, 
                    init_volume=10*3600, 
                    levels=(2000, 2120)),
              Basin(name='basin_2', 
                    volume=31*3600, 
                    num_states=41, 
                    init_volume=10*3600, 
                    levels=(1200, 1250))
              ]
    
    outflow = Outflow(outflow_level=600)
    
    turbines = [Turbine('turbine_1', 
                        max_power = 33e6,
                        base_load = 10e6,
                        efficiency=0.8, 
                        upper_basin=basins[0], 
                        lower_basin=basins[1]),
                Turbine('turbine_2', 
                        max_power = 15e6,
                        base_load =  7e6,
                        efficiency=0.8,
                        upper_basin=basins[1], 
                        lower_basin=outflow)
                ]
    
    actions = [ActionStanding(turbines[0]), 
               ActionPowerMin(turbines[0]),
               ActionPowerMax(turbines[0]),
               ActionStanding(turbines[1]), 
               ActionPowerMin(turbines[1]),
               ActionPowerMax(turbines[1])
               ]
    
    
    power_plant = PowerPlant(basins, turbines, actions) 
    
    pp_actions = power_plant.actions()
    

    constraints = [TurbineConstraint(turbines[0], '2020-07-29T00', '2020-07-29T02',
                                     name='test_0', margin_max=-2), 
                   TurbineConstraint(turbines[0], '2020-07-29T01', '2020-07-29T03',
                                     name='test_1', margin_max=-4), 
                   TurbineConstraint(turbines[1], '2020-07-29T02', '2020-07-29T03',
                                     name='test_2', margin_min=+2e6)
                   ]
    
    start_time = np.datetime64('2020-07-28T23')
    end_time =  np.datetime64('2020-07-29T05')
    
    constraints_series =  ConstraintsSeries(start_time, end_time, constraints)
        
    turbine_power = pp_actions[1].turbine_power(constraints_series[3])
    
    dt = 3600
    
    #make CoreActions
    def compute_core_action_series(power_plant, constraints_series, dt):
        unique_core_actions = {}
        core_action_series = []
        
        for contraints in constraints_series.normalized(turbines):
            
            key = tuple([tuple(constraint) for constraint in contraints.values()])
            
            if key not in unique_core_actions:
                
                core_actions = []
                for pp_action in power_plant.actions():
                    core_actions.append(
                        CoreAction(
                            pp_action.turbine_power(contraints), 
                            pp_action.basin_flow_rates(contraints), 
                            power_plant.basin_volumes(), 
                            power_plant.basin_num_states())
                        )
                            
                unique_core_actions[key] = core_actions
                        
            core_action_series.append(unique_core_actions[key])
            
        return core_action_series
                

    
    
                
        
        