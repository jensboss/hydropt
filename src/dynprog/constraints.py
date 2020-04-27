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

class TurbinePowerConstraint():
    def __init__(self, turbine, abs_power_min=None, abs_power_max=None, 
                 rel_power_min=None, rel_power_max=None, spinning=False):
        self.turbine = turbine
        self.abs_power_min = abs_power_min
        self.abs_power_max = abs_power_max
        self.rel_power_min = rel_power_min
        self.rel_power_max = rel_power_max
        
    def __eq__(self, other):
        return (self.turbine is other.turbine and
                self.abs_power_min == other.abs_power_min and
                self.abs_power_max == other.abs_power_max and
                self.rel_power_min == other.rel_power_min and
                self.rel_power_max == other.rel_power_max)
        
    def __hash__(self):
        return hash((self.turbine, self.abs_power_min, self.abs_power_max,
                     self.rel_power_min, self.rel_power_max))
    
    def __repr__(self):
        constraints = ''
        if self.abs_power_min is not None:
            constraints += f", abs_power_min={self.abs_power_min}"
        if self.abs_power_max is not None:
            constraints += f", abs_power_max={self.abs_power_max}"
        if self.rel_power_min is not None:
            constraints += f", rel_power_min={self.rel_power_min}"
        if self.rel_power_max is not None:
            constraints += f", abs_power_min={self.rel_power_max}"
            
        return f"{self.__class__.__name__}({self.turbine}{constraints})"
    
class Constraints():
    def __init__(self, constraints=None):
        if constraints is None:
            self.constraints = []
        else:
            self.constraints = constraints
        
    def append(self, constraint):
        self.constraints.append(constraint)
        
    def __repr__(self):
         return f"Constraints({self.constraints})"
    
class ConstrainedInterval():
    def __init__(self, start_time, end_time, constraint=None, turbine=None, 
                 abs_power_min=None, abs_power_max=None, rel_power_min=None, 
                 rel_power_max=None, spinning=False):
        
        self.start_time = np.datetime64(start_time)
        self.end_time = np.datetime64(end_time)
        
        if constraint is None:
            self.constraint = TurbinePowerConstraint(turbine, abs_power_min, 
                                                     abs_power_max, rel_power_min, 
                                                     rel_power_max, spinning)
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
                                                 rel_power_max=2e6)]
    
    print(constrained_intervals[0].constraint)
    
    start_time = '2020-04' 
    end_time =  '2020-05'
    print(ConstrainedIntervals(constrained_intervals).roll_out(start_time, end_time, 'h'))
    

