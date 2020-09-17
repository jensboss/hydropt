from hydropt.core import sparse
import torch
import numpy as np

from hydropt.utils import tensor_sparse_diags

class TestTensorSparseDiags():
    def test_square_matrix(self):
        offsets = [0, 1, -1]
        diags = [0.1, 0.9, 0.8]
        
        shape = (4, 4)
        
        diag_scipy = sparse.diags(diags, offsets=offsets, shape=shape)
        
        diag_torch = tensor_sparse_diags(diags, offsets=offsets, size=shape)
        
        assert np.all(diag_scipy.todense() == diag_torch.to_dense().numpy())
        
    def test_wide_matrix(self):
        offsets = [0, 3, -1]
        diags = [0.1, 0.9, 0.8]
        
        shape = (4, 8)
        
        diag_scipy = sparse.diags(diags, offsets=offsets, shape=shape)
        
        diag_torch = tensor_sparse_diags(diags, offsets=offsets, size=shape)
        
        assert np.all(diag_scipy.todense() == diag_torch.to_dense().numpy())
        
    def test_long_matrix(self):
        offsets = [0, 1, -2]
        diags = [0.1, 0.9, 0.8]
        
        shape = (9, 7)
        
        diag_scipy = sparse.diags(diags, offsets=offsets, shape=shape)
        
        diag_torch = tensor_sparse_diags(diags, offsets=offsets, size=shape)
        
        assert np.all(diag_scipy.todense() == diag_torch.to_dense().numpy())
        



if __name__ == '__main__':
    import timeit
    
    offsets = [0, 1, -2]
    diags = [0.1, 0.9, 0.8]        
    shape = (9, 7)
            
    t = timeit.timeit(lambda: tensor_sparse_diags(diags, offsets=offsets, size=shape),
                      number=100)
    print('time tensor_sparse_diags (in sec):', t)
    
    t = timeit.timeit(lambda: sparse.diags(diags, offsets=offsets, shape=shape),
                      number=100)
    print('time scipy sparse.diags (in sec):', t)
    
    
    
    
    
