import sys
sys.path.append(".")
from reid.datasets.grid import GRID
from reid.datasets.prid import PRID
from reid.datasets.ilids import iLIDS
from reid.datasets.viper import VIPeR


if __name__ == '__main__':
    root = '/data/datasets'
    grid = GRID(root + '/grid')
    viper = VIPeR(root + '/viper')
    prid = PRID(root + '/prid')
    ilids = iLIDS(root + '/ilids')
