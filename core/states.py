#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:14:02 2020

@author: Jens
"""


class StateSpace():
    def __init__(self, num_states):
        self.num_states = num_states
        
class BasinStates(StateSpace):
    def __init__(self, num_states, volume):
        super().__init__(self, num_states)
        self.volume = volume