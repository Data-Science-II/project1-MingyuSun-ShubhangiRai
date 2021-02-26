//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** @author  Shubhangi Rai
 *  @version 1.0
 *  @date    Feb 18, 2021
 *  @see     LICENSE (MIT style license file)
 */

package main.RegressionProject
import scalation.columnar_db.Relation
import scalation.util.banner
import scala.collection.mutable.Set
import scalation.linalgebra._
import scalation.analytics._
import scalation.plot.PlotM


//:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** Defining IllegalChoiceException class
*/
class IllegalChoiceException(s: String) extends Exception(s){}

class Exception1{
	@throws(classOf[IllegalChoiceException])
	def validate(choice: Int){
		if((choice < 0) || (choice > 11)) {
			throw new IllegalChoiceException("Invalid Choice.")
		}
	}
}

//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
/** The 'RegressionTest' object uses the defined cross-validation class, pre-defined 
* MatrixD and Regression classes to perform multiple regressions and subsequent analysis 
* on different numerical datasets, in the 'data' folder.  
*  > "sbt run" in the Scalation folder containing the build file to run the program.
* User gets two choices, once, to run on the dataset of his/her choice and again, to 
* choose the model to build the R2-RBar2-RCV2 graph on.
*/			
object RegressionTest extends App {

	// Method to implement the Simple Regression Model for R2-RBar2-RCV2 plot
	def regression_forward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Forward Selection... ")
		val rg_forward = new Regression (x, y)
		val (cols , rSq) = rg_forward.forwardSelAll()		
		println("Report", rg_forward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_forward.png")
		banner ("Successfully implemented Simple Regression with Forward Selection!")
	}
	
	def regression_backward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Backward Elimination... ")
		val rg_backward = new Regression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_backward.backwardElimAll()
		println("Report", rg_backward.analyze().report)
		println(rg_backward)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,rSq.dim1)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_backward.png")
		banner ("Successfully implemented Simple Regression with Backward Elimination!")
	}
	
	def regression_stepwise (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_stepwise = new Regression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_stepwise.stepRegressionAll()
		println("Report", rg_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_stepwise.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_lasso_forward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_lasso_forward = new LassoRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_lasso_forward.forwardSelAll()
		println("Report", rg_lasso_forward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_lasso_forward.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_lasso_backward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_lasso_backward = new LassoRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_lasso_backward.backwardElimAll()
		println("Report", rg_lasso_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_lasso_backward.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_lasso_stepwise (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_lasso_stepwise = new LassoRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_lasso_stepwise.stepRegressionAll()
		println("Report", rg_lasso_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_lasso_stepwise.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_ridge_forward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_ridge_forward = new RidgeRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_ridge_forward.forwardSelAll()
		println("Report", rg_ridge_forward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_ridge_forward.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_ridge_backward (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_ridge_backward = new RidgeRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_ridge_backward.backwardElimAll()
		println("Report", rg_ridge_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_ridge_backward.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def regression_ridge_stepwise (x: MatriD, y: VectorD)
	{
		banner ("Implementing Regression with Stepwise Selection... ")
		val rg_ridge_stepwise = new RidgeRegression (x, y)	// Instantiating a simple regression model
		val (cols , rSq ) = rg_ridge_stepwise.stepRegressionAll()
		println("Report", rg_ridge_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("regression_ridge_stepwise.png")
		banner ("Successfully implemented Simple Regression with Stepwise Selection!")
	}
	
	def quad_regression_forward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Quadratic Regression with Forward Selection... ")
		val rg_forward = new QuadRegression (x_initial, y)	// Instantiating a Quadratic regression model
		println("rg_forward",rg_forward)// Instantiating a simple regression model
		val (cols , rSq) = rg_forward.forwardSelAll()
		println("Report", rg_forward.analyze().report)
		
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quad_regression_forward.png")
		banner ("Successfully implemented Quadratic Regression with Forward Selection!")
	}
	
	def quad_regression_backward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Quadratic Regression with Backward Elimination... ")
		val rg_backward = new QuadRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_backward.backwardElimAll()
		println("Report", rg_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,rSq.dim1)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quad_regression_backward.png")
		banner ("Successfully implemented Quadratic Regression with Backward Elimination!")
	}
	
	def quad_regression_stepwise (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Quadratic Regression with Stepwise Selection... ")
		val rg_stepwise = new QuadRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_stepwise.stepRegressionAll()
		println("Report", rg_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quad_regression_stepwise.png")
		banner ("Successfully implemented Quadratic Regression with Stepwise Selection!")
	}
	
		def quadx_regression_forward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing QuadX Regression with Forward Selection... ")
		val rg_forward = new QuadXRegression (x_initial, y)	// Instantiating a Quadratic regression model
		println("rg_forward",rg_forward)// Instantiating a simple regression model
		val (cols , rSq) = rg_forward.forwardSelAll()
		println("Report", rg_forward.analyze().report)
		
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quadx_regression_forward.png")
		banner ("Successfully implemented QuadraticX Regression with Forward Selection!")
	}
	
	def quadx_regression_backward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing QuadX Regression with Backward Elimination... ")
		val rg_backward = new QuadXRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_backward.backwardElimAll()
		println("Report", rg_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,rSq.dim1)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quadx_regression_backward.png")
		banner ("Successfully implemented QuadX Regression with Backward Elimination!")
	}
	
	def quadx_regression_stepwise (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing QuadX Regression with Stepwise Selection... ")
		val rg_stepwise = new QuadXRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_stepwise.stepRegressionAll()
		println("Report", rg_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("quadx_regression_stepwise.png")
		banner ("Successfully implemented QuadX Regression with Stepwise Selection!")
	}
	
		def cubic_regression_forward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Cubic Regression with Forward Selection... ")
		val rg_forward = new CubicRegression (x_initial, y)	// Instantiating a Quadratic regression model
		println("rg_forward",rg_forward)// Instantiating a simple regression model
		val (cols , rSq) = rg_forward.forwardSelAll()
		println("Report", rg_forward.analyze().report)
		
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubic_regression_forward.png")
		banner ("Successfully implemented Cubic Regression with Forward Selection!")
	}
	
	def cubic_regression_backward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Cubic Regression with Backward Elimination... ")
		val rg_backward = new CubicRegression (x_initial, y)	// Instantiating a Cubic regression model
		val (cols , rSq ) = rg_backward.backwardElimAll()
		println("Report", rg_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,rSq.dim1)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubic_regression_backward.png")
		banner ("Successfully implemented Cubic Regression with Backward Elimination!")
	}
	
	def cubic_regression_stepwise (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing Cubic Regression with Stepwise Selection... ")
		val rg_stepwise = new CubicRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_stepwise.stepRegressionAll()
		println("Report", rg_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubic_regression_stepwise.png")
		banner ("Successfully implemented Cubic Regression with Stepwise Selection!")
	}
	
	def cubicx_regression_forward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing CubicX Regression with Forward Selection... ")
		val rg_forward = new CubicXRegression (x_initial, y)	// Instantiating a Quadratic regression model
		println("rg_forward",rg_forward)// Instantiating a simple regression model
		val (cols , rSq) = rg_forward.forwardSelAll()
		println("Report", rg_forward.analyze().report)
		
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubicx_regression_forward.png")
		banner ("Successfully implemented CubicX Regression with Forward Selection!")
	}
	
	def cubicx_regression_backward (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing CubicX Regression with Backward Elimination... ")
		val rg_backward = new CubicXRegression (x_initial, y)	// Instantiating a CubicX regression model
		val (cols , rSq ) = rg_backward.backwardElimAll()
		println("Report", rg_backward.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,rSq.dim1)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubicx_regression_backward.png")
		banner ("Successfully implemented CubicX Regression with Backward Elimination!")
	}
	
	def cubicx_regression_stepwise (x_initial: MatriD, y: VectorD)
	{
		banner ("Implementing CubicX Regression with Stepwise Selection... ")
		val rg_stepwise = new CubicXRegression (x_initial, y)	// Instantiating a Quadratic regression model
		val (cols , rSq ) = rg_stepwise.stepRegressionAll()
		println("Report", rg_stepwise.analyze().report)
		println(s"rSq = $rSq")
		val aic = rSq.sliceCol(3, 4)
		println("aic",aic)
		val k = cols.size
		val t = VectorD.range(1,k)
		new PlotM(t,rSq.t, Array ("R^2","R^2 bar", "R^2 cv"), "R^2 vs n for Regression", lines = true).saveImage("cubicx_regression_stepwise.png")
		banner ("Successfully implemented CubicX Regression with Stepwise Selection!")
	}
	
	
	
	def main(){
		// Giving user the choice to select from ten datasets, or to give data path to own CSV file
		println("-"*75)
		println (" Select dataset: \n\t 1. Auto MPG \n\t 2. Beijing PM2.5 Dataset \n\t 3. Concrete Compressive Strength Dataset \n\t 4. Real Estate Valuation Dataset \n\t 5. Parkinson's Tele Monitoring \n\t 6. Computer Hardware")
		println("\t 7. For other datasets, enter: /correct/path/to/data/csv")
		println("-"*75)
		
		val choice	 = scala.io.StdIn.readLine()
		// Exception, to alert if user choice is not between 1 and 11
		var e = new Exception1()
		try {
			e.validate(choice.toInt)
		} catch {
			case ex: Exception => println("Exception Occured : " + ex)
		}
						
		val filename = if(choice != "11") {choice + ".csv"} else {scala.io.StdIn.readLine()}  // Reads user's input for data path if user enters '11'
		val dataset = Relation (filename, "dataset", null, -1) 			// Saving CSV as a relation
		val column_names = dataset.colNames			// Array of column names in relation
		val num_cols = dataset.cols					// Number of columns in dataset

		// Implementation for Mean Imputation 		
		for(i <- 0 to (num_cols - 1)){
			val selected = dataset.sigmaS(column_names(i), (x) => x!="")	// Filtering rows which have a missing value, as a no entry, i.e, ""
			val v_selected = selected.toVectorS(column_names(i))			// Converting remaining elements in column into a vector
			val v_seld = v_selected.map((x) => x.toDouble)					// Converting each element in filtered column to Double data type 
			val mean_col = (v_seld.sum) / selected.count(column_names(i))	// Computing mean of filtered column elements
			dataset.update(column_names(i), mean_col.toString(), "") 		// Updating blank spaces with mean of column
		} 
		
		// Giving user choice to execute regression model of their choice
		println("-"*75)
		println ("Select model:\n\t 1. Regression with Forward Selection \n\t 2. Regression with Backward Elimination \n\t 3. Regression with Stepwise Selection \n\t 4. Lasso Regression with Forward Selection \n\t 5. Lasso Regression with Backward Elimination \n\t 6. Lasso Regression with Stepwise Selection \n\t 7. Ridge Regression with Forward Selection \n\t 8. Ridge Regression with Backward Elimination \n\t 9. Ridge Regression with Stepwise Selection \n\t 10. Quad Regression with Forward Selection \n\t 11. Quad Regression with Backward Elimination \n\t 12. Quad Regression with Stepwise Selection \n\t 13. QuadX Regression with Forward Selection \n\t 14. QuadX Regression with Backward Elimination \n\t 15. QuadX Regression with Stepwise Selection \n\t 16. Cubic Regression with Forward Selection \n\t 17. Cubic Regression with Backward Elimination \n\t 18. Cubic Regression with Stepwise Selection \n\t 19. CubicX Regression with Forward Selection \n\t 20. CubicX Regression with Backward Elimination \n\t 21. CubicX Regression with Stepwise Selection \n\t")
		println("-"*75)
		
		val model = scala.io.StdIn.readLine()
		if (model == "1") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_forward(x, y)	// Implementing Simple Regression Model when user's choice is '1'
		} else if (model == "2") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_backward(x, y)	// Implementing regression_backward Model when user's choice is '2'
		} else if (model == "3") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_stepwise(x, y)	// Implementing regression_stepwise when user's choice is '3'
		} else if (model == "4") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_lasso_forward(x, y)	// Implementing regression_lasso_forward Model when user's choice is '4'
		} else if (model == "5") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_lasso_backward(x, y)	// Implementing regression_lasso_backward when user's choice is '5'
		} else if (model == "6") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_lasso_stepwise(x, y)	// Implementing regression_lasso_stepwise Model when user's choice is '6'
		} else if (model == "7") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_ridge_forward(x, y)	// Implementing regression_ridge_forward Model when user's choice is '7'
		} else if (model == "8") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_ridge_backward(x, y)	// Implementing regression_ridge_backward when user's choice is '8'
		} else if (model == "9") {
			val (x_initial, y) = dataset.toMatriDD(1 until num_cols, 0) // Y vector is the first column of Relation
			val x = VectorD.one (x_initial.dim1) +^: x_initial	// Appending 1 column to x
			regression_ridge_stepwise(x, y)	// Implementing regression_ridge_stepwise when user's choice is '9'
		} else if (model == "10") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quad_regression_forward(x, y)	// Implementing quad_regression_forward Model when user's choice is '10'
		} else if (model == "11") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quad_regression_backward(x, y)	// Implementing quad_regression_backward Model when user's choice is '11'
		} else if (model == "12") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quad_regression_stepwise(x, y)	// Implementing quad_regression_stepwise Model when user's choice is '12'
		} else if (model == "13") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quadx_regression_forward(x, y)	// Implementing quadx_regression_forward Model when user's choice is '13'
		} else if (model == "14") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quadx_regression_backward(x, y)	// Implementing quadx_regression_backward Model when user's choice is '14'
		} else if (model == "15") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			quadx_regression_stepwise(x, y)	// Implementing quadx_regression_stepwise Model when user's choice is '15'
		} else if (model == "16") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubic_regression_forward(x, y)	// Implementing cubic_regression_forward Model when user's choice is '16'
		} else if (model == "17") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubic_regression_backward(x, y)	// Implementing cubic_regression_backward Model when user's choice is '17'
		} else if (model == "18") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubic_regression_stepwise(x, y)	// Implementing Quad Regression Model when user's choice is '18'
		} else if (model == "19") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubicx_regression_forward(x, y)	// Implementing cubicx_regression_forward when user's choice is '19'
		} else if (model == "20") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubicx_regression_backward(x, y)	// Implementing Quad Regression Model when user's choice is '20'
		} else if (model == "21") {
			val (x, y) = dataset.toMatriDD(1 until num_cols, 0)	// Y vector is the first column of Relation
			cubicx_regression_stepwise(x, y)	// Implementing Quad Regression Model when user's choice is '21'
		} 
	}

	main()
}
