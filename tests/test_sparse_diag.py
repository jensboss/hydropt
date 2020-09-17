from hydropt.core import sparse
import torch

if __name__ == '__main__':
    offsets = [0,1,-1]
    diags = [0.1,0.9,0.8]
    L = sparse.diags(diags, offsets=offsets, shape=(4,5))
    print(L.todense())
    L = sparse.diags(diags, offsets=offsets, shape=(5,4))
    print(L.todense())
    
    offset = offsets[0]
    
    shape=(5,4)
    
    def sparse_diags(diags, offsets, size):
                
        indices = []
        values = []
        
        for diag, offset in zip(diags, offsets):
            
            j = torch.arange(start=max(0,offset),
                            end=min(size[1], size[0]+offset),
                            dtype=torch.int64)
            
            i = torch.arange(start=max(0,-offset),
                            end=min(size[0], size[1]-offset),
                            dtype=torch.int64)    
            
            indices.append(torch.stack((i,j)))
            
            values.append(torch.FloatTensor([diag,]).repeat(i.size())) 
        
        return torch.sparse.FloatTensor(
            torch.cat(indices,1), 
            torch.cat(values), 
            size)
    
    
    print(sparse_diags(diags, offsets, shape).to_dense())
    
    
