import numpy as np
from hydropt.core import kron_index

def kron_index_ref(num_states, position):
    index = np.ones(1, dtype=np.int64)
    
    for k, num in enumerate(num_states):
        if k == position:
            index = np.kron(index, np.arange(num))
        else:
            index = np.kron(index, np.ones(num, dtype=np.int64))
            
    return index
    

class TestKronIndex():
    def test_position_zero(self):
        num_states = [2,3,4]
        position = 0
        assert np.all(kron_index(num_states, position) == kron_index_ref(num_states, position))
        
    def test_position_one(self):
        num_states = [2,3,4]
        position = 1
        assert np.all(kron_index(num_states, position) == kron_index_ref(num_states, position))
    
    
if __name__ == '__main__':
    import timeit
    
    num_states = [20,37,3,7]
    position = 0
        
    t = timeit.timeit(lambda: kron_index(num_states, position), number=100)
    print('time new kron_index (in sec):', t)
    
    t = timeit.timeit(lambda: kron_index_ref(num_states, position), number=100)
    print('time ref kron_index (in sec):', t)