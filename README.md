# project1-MingyuSun-ShubhangiRai

Group Members-  Shubhangi Rai and Mingyu Sun

Shubhangi Rai- 1) Project done in Scalation - Regression.scala
                    2) Stepwise Regression, Lasso and Ridge Regression done on Python -                            Stepwise_regression_Lasso_Ridge.py

Report done for the work done 

Mingyu Sun:       1) Project done in Scalation 
                    2) Forward Selection , Backward Selection and Stepwise Regression, done on Python

Report done for the work done.



Mingyu Sun- 

Download project1.scala and PredictorMat.scala, replace the PredictorMat.scala and add project1.scala under scalation_1.6/scalation_modeling/src/main/scala/scalation/analytics.
The whole project1 should be run inside scalation_1.6, with the following code

cd scalation_1.6 
$ ./build_all.sh 
cd scalation_modeling 
sbt
runMain scalation.analytics.the_object

Multiple Linear Regression is implemented in a separate object with the first column being 1, to run this case, use
runMain scalation.analytics.Regressionwith1

Multiple Linear Regression is also implemented together with Forward Selection, Backward Elimination, Stepwise Regression for comparison, with corresponding banners, so do Quad Regression, QuadX Regression, Cubic Regression, CubicX Regression, Ridge Regression, Lasso Regression.
For example, with 
runMain scalation.analytics.Regressionforward
both purely Multiple Linear Regression and Forward Selection for Multiple Linear Regression is run, but we didn't add 1s as the first column for Multiple Linear Regression. 
Similarly, with 
runMain scalation.analytics.QuadRegressionstepwise
both QuadX Regression and Stepwise Regression for QuadX Regression are run.
This follows the example of RegressionTest11 and other examples inside scalation_1.6.

To with other data sets, make sure the corresponding data files are under scalation_1.6//data/analytics, ( we provide some files for you to download ) then use
runMain scalation.analytics.dataset
To use data sets that are already coded in scalation, uncomment line 28-30 in project1.scala
//import ExampleBPressure._
//import ExampleConcrete._
//import ExampleBasketBall._

All objects for Multiple Linear Regression, Quad Regression, QuadX Regression, Cubic Regression, CubicX Regression, Ridge Regression, Lasso Regression with Forward Selection, Backward Elimination, Stepwise Regression or no feature selection can be found in project1.scala. stepwiseSelAll () and stepwiseSel can be found in PredictorMat.scala 

As neither Mingyu Sun nor Shubhangi Rai had a good idea of how to write code together, both of us finish the project with scalation. Both of us write some code with python too. However, we try to write the report together and choose other data sets together.


The python file consists of 3 parts, please commend out other parts when running one certain part. The first two are feature selection codes for the three methods from Mingyu Sun. The results can be obtained via 
forward_selection(X,y, sl)
backward_elimination(X,y, sl)
stepwise_selection(X,y, slin, slout)
We also provide an alternative way to do feature selection with SequentialFeatureSelector from mlxtend.feature_selection, though Mingyu didn't try this out due to the mlxtend package.

The third part was written by Shubhangi Rai

As for the report, the current report ( report_from_MingyuSun ) is from Mingyu Sun, Shubhangi Rai will add the analysis with AIC, she may also change some parts with Stepwise Regression or modify some other part. Shubhangi Rai will also add some other data sets.



Shubhangi Rai:

To access Scalation and Python Projects, Data and Report_ShubhangiRai,

Navigate to Regression_Shubhangi:

Data- .csv files for all the 5 data sets 

lib - jar files are present under this

src->  regression.scala for scala project source code
src->  stepwise_regression_lasso_ridge.py for python project source code


1. SCALA

Configuration to Setup:
    Install Scala Version 2.12.10
    Library for scalation are provided in folder lib(.jar)
            1. Modeling 
            2. Statistics
            3. Mathematics
            4. Database
    



How to run?

Download ProjectRegression Dataset is provided in folder data
Locate to regression.scala file
$ sbt > compile 
Or can use run regression.scala 
Libraries are already included and built  console will prompt to choose dataset first After choosing dataset -> select model to run




Data Sets used- Downloaded data sets from UCI Machine Learning Repository

1.AutoMPG
2.Concrete
3.Beijing PM2.5
4.Real Estate
5.Computer Hardware


Changes were made in classes:
-PredictorMat.scala for Stepwise regression and 
Fit.scala to compute AIC values.


Following Regression Techniques were implemented for Forward Selection, Backward Selection , Stepwise Selection

Multiple Linear Regression
Quadratic Regression
QuadX Regression
Cubic Regression
CubicX Regression 

 Lasso and Ridge Regression were also implemented as an alternative to Best Subset selection  Regression.



2. PYTHON:

Configuration to Setup:
        Install Python Version 3.7
        Libraries included and needed to be installed in Python
            1. Scikit learn
            2. Pandas
            3. NumPy

How to run:
    Open terminal
    Run jupyter notebook
    Open “.ipynb” file from jupyter notebook
    


My teammate Mingyu Sun implemented Forward Selection, Backward Selection.

I, Shubhangi Rai implemented stepwise regression for MLR , Quadratic, and Cubic Regression Techniques which could be run on the stepwise.ipynb file

Lasso and Ridge Regression were also implemented on Python which could be run on Stepwise_regression_Lasso_Ridge.py 



Report appended with work done by Shubhangi Rai 
