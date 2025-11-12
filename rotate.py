import torch

from hadamard_utils import get_hadK, matmul_hadU_cuda

ALL_ROTATIONS = {}

# from https://github.com/spcl/QuaRot/blob/main/fake_quant/rotation_utils.py
def random_orthogonal_matrix(size, dtype, device):
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q.to(dtype)

class Rotation:
    
    @staticmethod
    def get_rotation(rotation, size, dtype, device):
        device = str(device)
        if (size, dtype, device) in ALL_ROTATIONS:
            return ALL_ROTATIONS[(size, dtype, device)]
        else:
            rot = None
            if rotation == 'none':
                rot = NoRotation()
            elif rotation == 'random':
                rot = SimpleRotation(size, random_orthogonal_matrix(size, dtype=dtype, device=device))
            elif rotation == 'hadamard':
                rot = HadamardRotation(size, dtype, device)
            else:
                raise RuntimeError(f"Unsupported rotate {rotation}")
            ALL_ROTATIONS[(size, dtype, device)] = rot
            return rot

    def __init__(self):
        raise NotImplementedError

    def apply(self, tensor):
        raise NotImplementedError

    def apply_trans(self, tensor):
        raise NotImplementedError

class NoRotation(Rotation):
    def __init__(self):
        pass

    def apply(self, inp):
        return inp
    
    def apply_trans(self, inp):
        return inp

class SimpleRotation(Rotation):
    def __init__(self, size, Q):
        self.size = size
        self.Q = Q
    
    def apply(self, inp):
        assert inp.shape[-1] == self.size, f"Input shape {inp.shape} does not match rotation size {self.size}"
        return inp @ self.Q
    
    def apply_trans(self, inp):
        return inp @ self.Q.T

class HadamardRotation(Rotation):
    def __init__(self, size, dtype, device):
        self.HadK_mat, self.HadK = get_hadK(size)
        self.HadK_mat = self.HadK_mat.to(dtype=dtype, device=device)

    def apply(self, inp):
        return matmul_hadU_cuda(inp, self.HadK_mat, self.HadK)
    
    def apply_trans(self, inp):
        return matmul_hadU_cuda(inp, self.HadK_mat.T, self.HadK)
