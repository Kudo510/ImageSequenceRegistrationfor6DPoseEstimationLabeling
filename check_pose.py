import numpy as np
import torch
few = np.load('ruapc/2poseEst/few.npy')
negVec = np.load('ruapc/1TLESSObj_Fine/v1fine.npy')
print(f'few shape is {few.shape}')
print(f'negVec shape is {negVec.shape}')

corres_ = torch.load('ruapc/1Cors/224_posVec/2.pt')
print('PosVec' +str(corres_.shape))

import numpy as np

def cartesianproduct(arr):
    np_tr = np.asarray([[z0, y0, x0] for x0 in arr for y0 in arr for z0 in arr])
    return np_tr

arr = [0, 1, 2]
result = cartesianproduct(arr)
print(result)
