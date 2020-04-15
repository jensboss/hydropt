#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:14:02 2020

@author: Jens
"""
import numpy as np
import scipy.sparse as sparse

def kron_index(l, position):   
    L_combined = sparse.dia_matrix(np.ones((1,1)))
    for k in np.arange(len(l)):
        if k == position:
            L = sparse.diags(np.arange(l[k]), shape=(l[k],l[k]))
        else:
            L = sparse.eye(l[k])
        L_combined = sparse.kron(L_combined, L)     
    return L_combined.diagonal().astype(np.int64)


class BasisItem():
    def __init__(self, parent=None):
        self.parent=None


class DiscreteSpace():
    def __init__(self, items=None, size=None):
        self._items = None
        if items is None:
            items = [BasisItem() for i in range(size)]
        self.items = items
        
    @property
    def size(self):
        return len(self._items)
    
    @property
    def items(self):
        return self._items
    
    @items.setter
    def items(self, items):
        for items_item in items:
            items_item.parent = self
        self._items = items

         
class StateSpace(DiscreteSpace):
    pass
    
class Action(BasisItem):
    pass

class ActionSpace(DiscreteSpace):
    pass
        

class TensorProductSpace():
    def __init__(self, subspaces):
        self._subspaces = []
        self._index = {}
        self.size = 0
        self.subspaces = subspaces
        
    @property
    def subspaces(self):
        return self._subspaces
    
    @subspaces.setter
    def subspaces(self, subspaces):
        self._subspaces = subspaces
        self._index = {subspace:i for i, subspace in enumerate(subspaces)}
        self.size = len(self._subspaces)
        
    def subspace_sizes(self):
        return np.array([space.size for space in self._subspaces])
    
    def size(self):
        return np.prod(self.size)
    
    def subspace_index(self, subspace):
        return kron_index(self.size, self._index[subspace])
    
    @property
    def items(self):
        def get_product_items(subspaces):
            if len(subspaces) > 1:
                product_items = get_product_items(subspaces[1:])
                ret_product_items = list()
                for item in subspaces[0].items:
                    for items in product_items:
                        ret_product_items.append((item, *items))
                return ret_product_items
            else:
                return [(item,) for item in subspaces[0].items]
        return get_product_items(self._subspaces)
                
    
    
class Basin(StateSpace):
    def __init__(self, size, volume):
        super().__init__(size=size)
        self.volume = volume

    
class BasinSystem(TensorProductSpace):
    pass

class TurbineAction(Action):
    def __init__(self, flow_rate, parent=None):
        super().__init__(parent)
        self.flow_rate = flow_rate
        
    def __repr__(self):
        return f"TurbineAction({ self.flow_rate })"

class TurbineActions(ActionSpace):
    def __repr__(self):
        return f"TurbineActions({ self.items })"

class ActionSpace(TensorProductSpace):
    pass

        
if __name__ == '__main__':
    basin_1 = Basin(10, 100)
    basin_2 = Basin(20, 2000)
    basin_system = BasinSystem([basin_1, basin_2])
    turbine_act_1 = TurbineActions(items=[TurbineAction(flow_rate=0), 
                                              TurbineAction(flow_rate=10)])
    turbine_act_2 = TurbineActions(items=[TurbineAction(flow_rate=0), 
                                              TurbineAction(flow_rate=5)])
    act_space = ActionSpace([turbine_act_1, turbine_act_2])
    
    
    
    
    
    
    
    
    
    
    