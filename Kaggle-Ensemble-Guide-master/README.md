Kaggle-Ensemble-Guide
=====================

A combination of Model Ensembling methods that is extremely useful for increasing accuracy of Kaggle's submission.
For more information: http://mlwave.com/kaggle-ensembling-guide/

## Installation:

    $ pip install -r requirements.txt

## Example:


    $ python ./src/correlations.py ./samples/method1.csv ./samples/method2.csv
    Finding correlation between: ./samples/method1.csv and ./samples/method2.csv
    Column to be measured: Label
    Pearson's correlation score: 0.67898
    Kendall's correlation score: 0.66667
    Spearman's correlation score: 0.71053
    
    $ python ./src/kaggle_vote.py "./samples/method*.csv" "./samples/kaggle_vote.csv"
    $ python ./src/kaggle_vote.py "./samples/_*.csv" "./samples/kaggle_vote_weighted.csv" "weighted"
    $ python ./src/kaggle_rankavg.py "./samples/method*.csv" "./samples/kaggle_rankavg.csv"
    $ python ./src/kaggle_avg.py "./samples/method*.csv" "./samples/kaggle_avg.csv"
    $ python ./src/kaggle_geomean.py  "./samples/method*.csv" "./samples/kaggle_geomean.csv"

## Result:

    ==> ./samples/method1.csv <==
    ImageId,Label
    1,1
    2,0
    3,9
    4,9
    5,3
    
    ==> ./samples/method2.csv <==
    ImageId,Label
    1,2
    2,0
    3,6
    4,2
    5,3
    
    ==> ./samples/method3.csv <==
    ImageId,Label
    1,2
    2,0
    3,9
    4,2
    5,3
    
    ==> ./samples/kaggle_avg.csv <==
    ImageId,Label
    1,1.666667
    2,0.000000
    3,8.000000
    4,4.333333
    5,3.000000
    
    ==> ./samples/kaggle_rankavg.csv <==
    ImageId,Label
    1,0.25
    
