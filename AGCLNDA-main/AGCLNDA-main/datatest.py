import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args  # 导入参数配置
import scipy.sparse as sp
from Utils.TimeLogger import log  # 日志工具
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import numpy as np

predir = './Datasets/mydata1/'  # 数据集目录
trnfile = predir + 'train0'  # 训练数据文件路径
tstfile = predir + 'test0'   # 测试数据文件路径
print(trnfile)
with open(trnfile, 'rb') as fs:
    ret = (pickle.load(fs) != 0).astype(np.float32)
    # print(ret)
    # print(fs)

mat = ret

print(mat)
print(mat.shape)
print(mat.nnz)
