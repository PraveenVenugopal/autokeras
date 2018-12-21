from keras.datasets import mnist
from autokeras import ImageClassifier
from autokeras.constant import Constant

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape+(1,))
    x_test = x_test.reshape(x_test.shape+(1,))
    # Pass Grid search space as searcher_args
    searcher_args = {}
    searcher_args['search_space'] = { 'length' : [20, 30, 40], 'width' :  [64, 128, 256] }
    clf = ImageClassifier(path= '/home/venugopal/Documents/autotester', verbose=True, augment=False, searcher_args=searcher_args, search_type=Constant.GRID_SEARCH)
    clf.fit(x_train, y_train, time_limit=40 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)

    print(y * 100)
