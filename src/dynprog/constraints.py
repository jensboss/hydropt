#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:39:22 2020

@author: Jens
"""


import numpy as np

class AbsolutPowerConstraint():
    def __init__(self, turbine, abs_power_min=None, abs_power_max=None):
        self.turbine = turbine
        self.abs_power_min = abs_power_min
        self.abs_power_max = abs_power_max
        
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        else:     
            return (self.turbine is other.turbine and
                    self.abs_power_min == other.abs_power_min and
                    self.abs_power_max == other.abs_power_max)
        
    def __hash__(self):
        return hash((self.turbine, self.abs_power_min, self.abs_power_max))
    
    def __repr__(self):
        constraints = ''
        
        if self.abs_power_min is not None:
            constraints += f", abs_power_min={self.abs_power_min}"
        if self.abs_power_max is not None:
            constraints += f", abs_power_max={self.abs_power_max}"
            
        return f"{self.__class__.__name__}({self.turbine}{constraints})"
    
    def copy(self):
        return self.__class__(self.turbine, self.abs_power_min, self.abs_power_max)


def none_func(a, b, func):
    if a is not None and b is not None:
        return func(a, b)
    elif a is None and b is not None:
        return b
    elif a is not None and b is None:
        return a
    else:
        return None
    
def none_min(a, b):
    return none_func(a, b, min)

def none_max(a, b):
    return none_func(a, b, max)
    
class Constraints():
    def __init__(self, constraints=None):
        self.constraints = dict()
        if constraints is not None:
            for constraint in constraints:
                self.append(constraint)
        
    def append(self, constraint):
        if constraint.turbine in self.constraints:
            known_constraint = self.constraints[constraint.turbine]
            
            known_constraint.abs_power_min = none_max(known_constraint.abs_power_min, 
                                                 constraint.abs_power_min)
            known_constraint.abs_power_max = none_min(known_constraint.abs_power_max, 
                                                 constraint.abs_power_max)
        else:
            self.constraints[constraint.turbine] = constraint.copy()
            
    def __getitem__(self, key):
        return self.constraints[key]
        
    def __eq__(self, other):
        return list(self.constraints.values()) == list(other.constraints.values())
    
    def __hash__(self):
        return hash(tuple(self.constraints.values()))
        
    def __repr__(self):
         return f"Constraints({list(self.constraints.values())})"

    
class ConstrainedInterval():
    def __init__(self, start_time, end_time, constraint=None, turbine=None, 
                 abs_power_min=None, abs_power_max=None):
        
        self.start_time = np.datetime64(start_time)
        self.end_time = np.datetime64(end_time)
        
        if constraint is None:
            self.constraint = AbsolutPowerConstraint(turbine,
                                                     abs_power_min,
                                                     abs_power_max)
        else:
            self.constraint = constraint
            
    def sample_interval(self, sampling_time):
        start_time = np.datetime64(self.start_time, sampling_time)
        end_time = np.datetime64(self.end_time, sampling_time)
        return np.arange(start_time, end_time)
    
    def intersect(self, time_range):
        mask = (self.start_time <= time_range) &  (time_range < self.end_time)
        return time_range[mask]
        
    def __contains__(self, time):
        return self.start_time <= np.datetime64(time) < self.end_time
    
        
class ConstrainedIntervals():
    def __init__(self, constrained_intervals=None):
        if constrained_intervals is None:
            self.constrained_intervals = []
        else:
            self.constrained_intervals = constrained_intervals
        
    def roll_out(self, time_range):
        time_index = dict()
        for k, time in enumerate(time_range):
            time_index[time] = k
        
        rolled_out_constraints = [Constraints() for time in time_range]
        
        for constrainted_interval in self.constrained_intervals:
            times_constraint = constrainted_interval.intersect(time_range)
            for time in times_constraint:
                ind = time_index[time]
                rolled_out_constraints[ind].append(constrainted_interval.constraint)
           
        constraints_cache = dict() 
        constraints = dict()
        constraints_ids = list()
        ctr = 0
        for consts in rolled_out_constraints:
            if not consts in constraints_cache:
                constraints_cache[consts] = ctr
                constraints[ctr] = consts
                ctr += 1
            constraints_ids.append(constraints_cache[consts])
            
        return constraints_ids, constraints
    
    
if __name__ == '__main__':
    from dynprog.model import Turbine, Basin, Outflow, ActionStanding, \
        ActionMin, ActionMax, PowerPlant
        
    from dynprog.scenarios import Scenario, Underlyings
    
    basins = [Basin(name='basin_1', 
                    volume=81, 
                    num_states=81, 
                    init_volume=10, 
                    levels=(2000, 2120)),
              Basin(name='basin_2', 
                    volume=31, 
                    num_states=41, 
                    init_volume=10, 
                    levels=(1200, 1250))
              ]
    
    outflow = Outflow(outflow_level=600)
    
    turbines = [Turbine('turbine_1', 
                        max_power = 33000000.0,
                        base_load = 10000000.0,
                        efficiency=0.8, 
                        upper_basin=basins[0], 
                        lower_basin=basins[1],
                        actions=[ActionStanding(), ActionMin(), ActionMax()]),
                Turbine('turbine_2', 
                        max_power = 15000000.0,
                        base_load =  7000000.0,
                        efficiency=0.8,
                        upper_basin=basins[1], 
                        lower_basin=outflow,
                        actions=[ActionStanding(), ActionMin(), ActionMax()])
                ]
    
    power_plant = PowerPlant(basins, turbines)    
    
    
    def date_range(start_time, end_time, sampling_time=None):
        if sampling_time is None:
            start_time = np.datetime64(start_time)
            end_time = np.datetime64(end_time)
        else:
            start_time = np.datetime64(start_time, sampling_time)
            end_time = np.datetime64(end_time, sampling_time)
            
        return np.arange(start_time, end_time)
        
    
    start_time = '2020-04-01T00' 
    end_time =  '2020-05'
    time = date_range(start_time, end_time)
    
    n_steps = len(time)
    hpfc = 10*(np.sin(2*np.pi*2*np.arange(n_steps)/n_steps) + 1)
    inflow = 0.8*np.ones((n_steps,2))
    
    underlyings = Underlyings(time, hpfc, inflow)
    
    
    constrained_intervals = ConstrainedIntervals(
        [ConstrainedInterval('2020-04-27',
                             '2020-04-28',
                             turbine=turbines[0],
                             abs_power_max=20e6),
         ConstrainedInterval('2020-04-26',
                             '2020-04-27T05',
                             turbine=turbines[0],
                             abs_power_min=1.5e6)]
        )
    
    constraints_ids, constraints = constrained_intervals.roll_out(time)
    
    print(constraints_ids, constraints)
    
    scenario = Scenario(power_plant, underlyings, ConstrainedIntervals, name='base')
    
    power_plant_actions = scenario.power_plant.actions()
    
    print(power_plant_actions.turbine_power(constraints[constraints_ids[0]]))

