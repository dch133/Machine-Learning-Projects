'''
Sample class to be used for testing implementations with incomplete models
'''

from model import Model

class Test(Model):

    def __init__(self):
        super().__init__()

    def fit(self, X, y, params=[]):
        print("Ran fit")
        return

    def predict(self, X, y):
        print("Ran predict")
        return 1

    def evaluate_acc(self, X, y):
        print("Ran evaluate_acc")
        return 50.0


if __name__ == '__main__':
    test = Test()
    print("Starting program")
    test.fit(1,2,3)
