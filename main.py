import pickle
import numpy as np

from saver_loader import DataSaver, DataLoader



path = "test_path"
saver = DataSaver("test_path", "sifnf")

data1 = [np.array([[0,1,3], [1,4,2]]), np.zeros((2,3,6), dtype=np.float32)]
saver.save(data1)
data2 = [np.zeros((3,4,5), dtype=np.int64), np.ones((2,3,6), dtype=np.bool)]
saver.save(data2)


loader = DataLoader(path)

for i in range(3):
    out = loader.getItem()
    for o in out:
        print(o.shape, o.dtype, end=" ")
    print()

