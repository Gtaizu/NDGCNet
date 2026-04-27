import sys
import scipy.sparse.coo as coo_module
sys.modules['scipy.sparse._coo'] = coo_module

import pickle
with open('你的pkl文件路径.pkl', 'rb') as f:
    data = pickle.load(f)

# 后续就可以正常用 data 了
