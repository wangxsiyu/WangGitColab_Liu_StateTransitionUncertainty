import torch
import numpy as np
class W_NN_initialize():
    def init_ortho2D(weights, scale = 1.):
        """ PyTorch port of ortho_init from baselines.a2c.utils """
        shape = tuple(weights.size())
        flat_shape = shape[1], shape[0]

        a = torch.tensor(np.random.normal(0., 1., flat_shape))

        u, _, v = torch.svd(a)
        t = u if u.shape == flat_shape else v
        t = t.transpose(1,0).reshape(shape).float()

        return scale * t
