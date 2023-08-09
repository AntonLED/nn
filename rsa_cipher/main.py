from DataManager import DataManager
import numpy as np


ROOT = "/home/anton/nn/rsa_cipher/datasets/"

dm = DataManager(nbits=256)

# dm.generate_dataset("test", 100)

print(np.genfromtxt(ROOT + "test_input.txt").shape)

