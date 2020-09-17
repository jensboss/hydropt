import numpy as np
import torch

from hydropt.utils import tensor_kron


class TestTensorKron():
    def test_random_square_matrices(self):
        m = 5
        A_array = np.random.rand(m,m)
        B_array = np.random.rand(m,m)
        
        A_tensor = torch.tensor(A_array.tolist(), dtype=torch.float64)
        B_tensor = torch.tensor(B_array.tolist(), dtype=torch.float64)
        
        kron_numpy = np.kron(A_array, B_array)
        kron_tensor = tensor_kron(A_tensor, B_tensor).numpy()
        
        assert np.all(kron_numpy == kron_tensor)
        
    def test_random_matrices_one(self):
        n = 5
        m = 3
        A_array = np.random.rand(n,m)
        B_array = np.random.rand(n,m)
        
        A_tensor = torch.tensor(A_array.tolist(), dtype=torch.float64)
        B_tensor = torch.tensor(B_array.tolist(), dtype=torch.float64)
        
        kron_numpy = np.kron(A_array, B_array)
        kron_tensor = tensor_kron(A_tensor, B_tensor).numpy()
        
        assert np.all(kron_numpy == kron_tensor)
        
    def test_random_matrices_two(self):
        n = 3
        m = 7
        A_array = np.random.rand(n,m)
        B_array = np.random.rand(n,m)
        
        A_tensor = torch.tensor(A_array.tolist(), dtype=torch.float64)
        B_tensor = torch.tensor(B_array.tolist(), dtype=torch.float64)
        
        kron_numpy = np.kron(A_array, B_array)
        kron_tensor = tensor_kron(A_tensor, B_tensor).numpy()
        
        assert np.all(kron_numpy == kron_tensor)
   
    
if __name__ == '__main__':
    import timeit
    
    n = 20
    m = 50
    A_array = np.random.rand(n,m)
    B_array = np.random.rand(n,m)
    
    A_tensor = torch.tensor(A_array.tolist(), dtype=torch.float64)
    B_tensor = torch.tensor(B_array.tolist(), dtype=torch.float64)
        
    t = timeit.timeit(lambda: tensor_kron(A_tensor, B_tensor), number=100)
    print('time tensor_kron (in sec):', t)
    
    t = timeit.timeit(lambda: np.kron(B_tensor, B_array), number=100)
    print('time numpy kron (in sec):', t)
    
    
    