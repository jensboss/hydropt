import torch

def tensor_kron(t1: torch.tensor, t2: torch.tensor):
    """Compute the Kronecker product between two tensors.
    """
    
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def tensor_sparse_diags(diags, offsets, size):
    """Construct sparse diagonal tensor of dtype float64.
    """
    
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
        
        values.append(torch.DoubleTensor([diag,]
                                        ).repeat(i.size())) 
    
    return torch.sparse.FloatTensor(torch.cat(indices,1), 
                                    torch.cat(values), 
                                    size)