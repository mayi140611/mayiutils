#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: scikit_surprise_wrapper.py
@time: 2019/3/13 12:22

https://pypi.org/project/scikit-surprise/#description
pip install scikit-surprise
scikit-surprise-1.0.6
"""
from surprise import SVD, KNNWithMeans
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate


if __name__ == '__main__':
    mode = 3
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
    # Load the movielens-100k dataset (download it if needed).
    # data = Dataset.load_builtin('ml-100k')
    if mode == 0:
        print(type(data))#<class 'surprise.dataset.DatasetAutoFolds'>
        print(data)#<surprise.dataset.DatasetAutoFolds object at 0x0000024F8F511AC8>
    if mode == 1:
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