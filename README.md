# project-bgoldman-kgriesman

Ben and Kendra: 12/3/19 (1 hr)
  - Read in dataset and separated data and labels
  - Filled in missing values with the mean
  - Ran KNN with the data
  - Still need to split test and train data
  - Data: https://archive.ics.uci.edu/ml/datasets/Ozone+Level+Detection onehr.data

Kendra: 12/5/19 (1 hr)
  - Split train and test data (80:20)
  - Set up confusion matrix for KNN

Ben: 12/5/19 (1 hr)
  - added random seed to split so that it split same way everytime
  - Professor suggested we try adaboost because very uneven number of labels
  - started SVM

Ben and Kendra: 12/10/19 (1.5 hrs)
  - Tried under and oversampling data
  - Tested various hyperparameters for SVM and AdaBoost
  - Added ROC curve for AdaBoost

Kendra: 12/12/19 (1 hr)
  - Found most important features for AdaBoost
  - RH50 (6.4%) Relative humidity at 500 hpa level (roughly at 5500 m height)
  - U70 (5.6%) U wind - east-west direction wind at 700 hpa level (roughly 3100 m height)
  - WSR20 (4%) Wind speed at hour 20
