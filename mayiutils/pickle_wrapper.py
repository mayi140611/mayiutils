import pickle


class PickleWrapper(object):

    @classmethod
    def loadFromFile(self, file, mode='rb'):
        with open(file, mode) as f:
            return pickle.load(f)
    
    @classmethod
    def dump2File(self, o, file, mode='wb'):
        '''
        把目标对象序列化到文件
        :param o: 目标对象
        :param file: 文件
        :param mode:
        :return:
        '''
        with open(file, mode) as f:
            pickle.dump(o, f)