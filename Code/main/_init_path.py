import os.path as osp
import sys
 

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


proj_dir = osp.abspath(osp.join(osp.dirname(__file__), osp.pardir))
add_path(proj_dir)
