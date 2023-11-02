import numpy as np

class Neuralnetwork:
    def __init__(self):
        self.weights = np.array([np.random.randn() for i in range(4)])
        self.bias = np.random.randn()


    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def make_prediction(self,):
        pass


array = [1,2,3,4,5]

print(sum([x**2 for x in array]))

(x for x in range(5))

{a:b for (a,b) in enumerate((range(5)))}

[[1,2],[3,4],[5,6]]