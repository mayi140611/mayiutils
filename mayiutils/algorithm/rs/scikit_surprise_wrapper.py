#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: scikit_surprise_wrapper.py
@time: 2019/3/13 12:22

https://pypi.org/project/scikit-surprise/#description
https://surprise.readthedocs.io/en/stable/getting_started.html
pip install scikit-surprise
scikit-surprise-1.0.6
"""
from surprise import SVD, KNNWithMeans, KNNBasic, NormalPredictor, evaluate
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise import accuracy


class SurpriseWrapper:
    """
    surprise中常用的算法 http://www.360doc.com/content/17/1218/21/40769523_714320532.shtml

    surprise常用的数据结构
    1、surprise.dataset.Dataset：主要的作用是载入数据集
    2、surprise.Trainset：
        https://surprise.readthedocs.io/en/stable/trainset.html#surprise.Trainset
        A trainset contains all useful data that constitutes a training set.
        Trainsets are different from Datasets. You can think of a Datasets as the raw data,
        and Trainsets as higher-level data where useful methods are defined.
        Also, a Datasets may be comprised of multiple Trainsets (e.g. when doing cross validation).
        Trainset可以Dataset生成
            Dataset.folds() 已废弃
            DatasetAutoFolds.build_full_trainset()
            surprise.model_selection.train_test_split(Dataset)
        Trainset常用属性
            ur：The users ratings
                This is a dictionary containing lists of tuples of the form (item_inner_id, rating).
                The keys are user inner ids.
            ir：The items ratings
                This is a dictionary containing lists of tuples of the form (user_inner_id, rating).
                The keys are item inner ids.
            n_users
            n_items
            n_ratings
            rating_scale：评分范围
            global_mean：所有评分的均值
            all_items()：Generator function to iterate over all items.
            all_ratings()

    """
    def __init__(self):
        pass


if __name__ == '__main__':
    mode = 4

    if mode == 4:
        """
        官方提供的get-started
        """
        submode = 403

        if submode == 401:
            """
            Automatic cross-validation
            """
            data = Dataset.load_builtin('ml-100k')
            # Use the famous SVD algorithm.
            algo = SVD()
            # Run 5-fold cross-validation and print results.
            cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=5, verbose=True)
            """
                              Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
            RMSE (testset)    0.9346  0.9331  0.9364  0.9324  0.9353  0.9344  0.0015  
            MAE (testset)     0.7349  0.7371  0.7388  0.7318  0.7388  0.7363  0.0026  
            FCP (testset)     0.7024  0.6981  0.7051  0.7001  0.7014  0.7014  0.0024  
            Fit time          5.91    5.88    6.01    6.00    5.78    5.92    0.08    
            Test time         0.21    0.19    0.20    0.20    0.18    0.20    0.01  
            """
        elif submode == 402:
            """
            Train-test split and the fit() method            
            """
            # Load the movielens-100k dataset (download it if needed),
            data = Dataset.load_builtin('ml-100k')

            # sample random trainset and testset
            # test set is made of 25% of the ratings.
            trainset, testset = train_test_split(data, test_size=.25)
            print(trainset.rating_scale)#(1, 5)
            # We'll use the famous SVD algorithm.
            algo = SVD()

            # Train the algorithm on the trainset, and predict ratings for the testset
            algo.fit(trainset)
            predictions = algo.test(testset)

            # Then compute RMSE
            accuracy.rmse(predictions)#RMSE: 0.9430
        elif submode == 403:
            """
            Train on a whole trainset and the predict() method
            """
            # Load the movielens-100k dataset
            data = Dataset.load_builtin('ml-100k')

            # Retrieve the trainset.
            trainset = data.build_full_trainset()

            # Build an algorithm, and train it.
            algo = KNNBasic()
            algo.fit(trainset)

            uid = str(196)  # raw user id (as in the ratings file). They are **strings**!
            iid = str(302)  # raw item id (as in the ratings file). They are **strings**!

            # get a prediction for specific users and items.
            # The predict() uses raw ids!
            pred = algo.predict(uid, iid, r_ui=4, verbose=True)
            """
            Computing the msd similarity matrix...
            Done computing similarity matrix.
            user: 196        item: 302        r_ui = 4.00   est = 4.06   {'actual_k': 40, 'was_impossible': False}
            """
        elif submode == 404:
            #载入自己的数据集
            reader = Reader(line_format='user item rating', sep=',')
            filepath = '../../../tmp/doc.csv'
            data = Dataset.load_from_file(filepath, reader)
            print(data.raw_ratings)
            trainset = data.build_full_trainset()
            print(type(trainset))#<class 'surprise.trainset.Trainset'>
            print(trainset.n_items)#404
            print(trainset.n_users)#404
            data.split(n_folds=5)

    if mode == 2:
        # Use the famous KNNWithMeans algorithm.
        algo = KNNWithMeans()
        # Run 5-fold cross-validation and print results.
        cross_validate(algo, data, measures=['RMSE', 'MAE', 'FCP'], cv=5, verbose=True)
        """
        Computing the msd similarity matrix...
        Done computing similarity matrix.
        Computing the msd similarity matrix...
        Done computing similarity matrix.
        Computing the msd similarity matrix...
        Done computing similarity matrix.
        Computing the msd similarity matrix...
        Done computing similarity matrix.
        Computing the msd similarity matrix...
        Done computing similarity matrix.
        Evaluating RMSE, MAE, FCP of algorithm KNNWithMeans on 5 split(s).
        
                          Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
        RMSE (testset)    0.9514  0.9573  0.9517  0.9501  0.9455  0.9512  0.0038  
        MAE (testset)     0.7510  0.7503  0.7502  0.7502  0.7453  0.7494  0.0021  
        FCP (testset)     0.6973  0.7100  0.6979  0.7007  0.6996  0.7011  0.0046  
        Fit time          0.67    0.72    0.72    0.73    0.67    0.70    0.03    
        Test time         4.39    4.42    4.42    4.31    4.19    4.35    0.09 
        """