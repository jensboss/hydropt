#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:32:53 2020

@author: Jens
"""


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
                 power_max=None, power_min=None, 
                 margin_max=None, margin_min=None):
        
        self.turbine = turbine
        self.time_start = np.datetime64(time_start)
        self.time_end = np.datetime64(time_end)
        
        self.name = name
        
        self.power_max = power_max
        self.power_min = power_min
        self.margin_max = margin_max
        self.margin_min = margin_min
        
    def update(self, other):
        
        if isinstance(other, self.__class__):
            if not self.turbine == other.turbine:
                raise ValueError('Turbines not match.')
            other = tuple(other)[1:]
            
        self.power_max = minimum(self.power_max, other[0])
        self.power_min = maximum(self.power_min, other[1])
        
        self.margin_max = minimum(self.margin_max, other[2])
        self.margin_min = maximum(self.margin_min, other[3])
            
        
    def __add__(self, other):
        if self.turbine is not other.turbine:
            raise ValueError("Cannot sum constraints of different turbines. "
                             f"{self.turbine}, {other.turbine}")
            
        if self.name and other.name:
            name = '+'.join((self.name,other.name))
        else:
            name = ''
            
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
            
                
    
def make_constraint_series(time_start, time_end, constraints):
        
    time_start = np.datetime64(time_start)
    time_end =  np.datetime64(time_end)
    
    time = np.arange(start_time, end_time)
   
    

    indices = np.arange(len(time))
    
    series = [{} for i in indices]
    
    for constraint in constraints:
        ind = indices[(time>=constraint.time_start) & (time<constraint.time_end)]
        
        for i in ind:
            
            const_dict = series[i]
            
            if constraint.turbine in const_dict:
                const_dict[constraint.turbine] += constraint
            else:
                const_dict[constraint.turbine] = constraint
                
    return series    

            
if __name__ == '__main__':
    from dynprog.model import Basin, Outflow, Turbine, PowerPlant
    from dynprog.action import ActionStanding, ActionPowerMin, ActionPowerMax


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
    
    pp_actions = power_plant.actions().power_plant_actions
    

    constraints = [TurbineConstraint(turbines[0], '2020-07-29T00', '2020-07-29T02',
                                     name='test_0', margin_max=-2), 
                   TurbineConstraint(turbines[0], '2020-07-29T01', '2020-07-29T03',
                                     name='test_1', margin_max=-4), 
                   TurbineConstraint(turbines[1], '2020-07-29T02', '2020-07-29T03',
                                     name='test_2', margin_max=-4)
                   ]
    
    start_time = np.datetime64('2020-07-28T23')
    end_time =  np.datetime64('2020-07-29T05')
    
    series =  make_constraint_series(start_time, end_time, constraints)
        
    print(series)
                
            
            
                
        
        