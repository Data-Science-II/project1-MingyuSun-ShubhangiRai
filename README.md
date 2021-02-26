# project1-MingyuSun-ShubhangiRai
Download analytics and substitute the original analytics under scalation_1.6/scalation_modeling/src/main/scala/scalation with this. Alternatively you can only download project1.scala and PredictorMat.scala, replace the PredictorMat.scala and add project1.scala under /analytics.
The whole project1 should be run inside scalation_1.6, with the following code

cd scalation_1.6 
$ ./build_all.sh 
cd scalation_modeling 
sbt
runMain scalation.analytics.the_object

Multiple Linear Regression is implemented in a separate object with first column being 1, to run this case, use
runMain scalation.analytics.Regressionwith1

Multiple Linear Regression is also implemented together with Forward Selection, Backward Elimination, Stepwise Regression for comparison, with correspodning banners, so do Quad Regression, QuadX Regression, Cubic Regression, CubicX Regression, Ridge Regression, Lasso Regression.
For example, with 
runMain scalation.analytics.Regressionforward
both purly Multiple Linear Regression and Forward Selection for Multiple Linear Regression are run, but we didn't add 1s as the first column for Multiple Linear Regression. 
Similarly, with 
runMain scalation.analytics.QuadRegressionstepwise
both QuadX Regression and Stepwise Regressionn for QuadX Regression are run.
This follows the example of RegressionTest11 and other examples inside scalation_1.6.

To with other data sets, use
runMain scalation.analytics.dataset
To use data sets that are already coded in scalation, uncommend line 28-30 in project1.scala
//import ExampleBPressure._
//import ExampleConcrete._
//import ExampleBasketBall._

All objects for Multiple Linear Regression, Quad Regression, QuadX Regression, Cubic Regression, CubicX Regression, Ridge Regression, Lasso Regression with Forward Selection, Backward Elimination, Stepwise Regression or no feature selection can be found in project1.scala. stepwiseSelAll () and stepwiseSel can be found in PredictorMat.scala 

As neither Mingyu Sun nor Shubhangi Rai had good idea of how to write code together, both of us finish the project with scalation. Both of us write some code with python too. However, we try to write the report together and choose other data sets together.


The python file consists of 3 part, please commend out other parts when running one certain part. First two are feature selection codes for the three method from Mingyu Sun. The results can be obtained via 
forward_selection(X,y, sl)
backward_elimination(X,y, sl)
stepwise_selection(X,y, slin, slout)
We also provide an alternative way to do feature selection with SequentialFeatureSelector from mlxtend.feature_selection, though Mingyu didn't try this out due to the mlxtend package.

The Third part was written by Shubhangi Rai
