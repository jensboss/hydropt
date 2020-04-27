#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 12:39:22 2020

@author: Jens
"""


import numpy as np




class TimeIndex():
    def __init__(self, start_time, end_time, ts):
        self._time_index = None
        self._time_index_start = start_time
        self._time_index_end = end_time
        self._time_index_ts = ts
    
    @property
    def time_index(self):
        if self._time_index is None:
            start_time = np.datetime64(self._time_index_start, self._time_index_ts)
            end_time = np.datetime64(self._time_index_end, self._time_index_ts)
            sampled_time = np.arange(start_time, end_time)
            time_index = dict()
            for k, time_sample in enumerate(sampled_time):
                time_index[time_sample] = k
            self._time_index = time_index
            
        return self._time_index
    
    def __contains__(self, time):
        return (time in self.time_index)


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


def non_func(a, b, func):
    if a is not None and b is not None:
        return func(a, b)
    elif a is None and b is not None:
        return b
    elif a is not None and b is None:
        return a
    else:
        return None
    
def none_min(a, b):
    return non_func(a, b, min)

def none_max(a, b):
    return non_func(a, b, max)
    
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
        

    def __contains__(self, time):
        return self.start_time <= np.datetime64(time) < self.end_time
    
        
class ConstrainedIntervals():
    def __init__(self, constrained_intervals):
        self.constrained_intervals = constrained_intervals
        
    def roll_out(self, start_time, end_time, sampling_time):
        start_time = np.datetime64(start_time, sampling_time)
        end_time = np.datetime64(end_time, sampling_time)
        
        time_range = np.arange(start_time, end_time)
        
        time_index = dict()
        for k, time in enumerate(time_range):
            time_index[time] = k
        
        rolled_out_constraints = [Constraints() for time in time_range]
        
        for constrainted_interval in self.constrained_intervals:
            times_constraint = constrainted_interval.sample_interval(sampling_time)
            for time in times_constraint:
                k = time_index[time]
                if rolled_out_constraints[k] is None:
                    rolled_out_constraints[k] = list()
                rolled_out_constraints[k].append(constrainted_interval.constraint)
            
        return rolled_out_constraints
    
    
if __name__ == '__main__':
    from dynprog.model import Turbine, Basin, Outflow, ActionStanding, \
        ActionMin, ActionMax
    
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

    
    constrained_intervals = [ConstrainedInterval('2020-04-27', 
                                                 '2020-04-28', 
                                                 turbine=turbines[0],
                                                 abs_power_max=20e6),
                             ConstrainedInterval('2020-04-26', 
                                                 '2020-04-27T05', 
                                                 turbine=turbines[0],
                                                 abs_power_min=1.5e6)]
    
    print(constrained_intervals[0].constraint)
    
    start_time = '2020-04' 
    end_time =  '2020-05'
    print(ConstrainedIntervals(constrained_intervals).roll_out(start_time, end_time, 'h'))
    

