package scalation.analytics

import scala.collection.mutable.Set
import scala.util.control.Breaks.{break, breakable}

import scalation.linalgebra._
import scalation.math.noDouble
import scalation.plot.{Plot, PlotM}
import scalation.random.CDF.studentTCDF
import scalation.stat.Statistic
import scalation.stat.StatVector.corr
import scalation.util.banner
import scalation.util.Unicode.sub

import Fit._

import scala.collection.mutable.Set

import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD}
import scalation.linalgebra.VectorD.one
import scalation.math.double_exp
import scalation.util.banner

import MatrixTransform._
import RegTechnique._


//import ExampleBPressure._
//import ExampleConcrete._
//import ExampleBasketBall._

/** check other data sets from UCI Machine Learning Repository.
 * after checking the whole dataset, x, y can be assigned as val (x, y) = data_tab.toMatriDD (? to ?, ?)
 * if use Concrete or basketball data set with import above, use val (x, y) = pullResponse (xy) to assign x and y
 * if use BPressure data set with import above, x, y are already available
 */
object dataset extends App {

  import scalation.columnar_db.Relation

  import scalation.columnar_db.Relation
  //
  banner("check dataset")
  banner("diabetes relation")
  val data_tab = Relation(BASE_DIR + "diabetes.csv", "diabetes", null, -1)
  data_tab.show()

//  val (x, y) = data_tab.toMatriDD (1 to 6, 0)
//  println (s"x = $x")
//  println (s"y = $y")

}

object Regressionwith1 extends App {

  import scalation.columnar_db.Relation
  //
  banner("auto_mpg relation")
  val auto_tab = Relation(BASE_DIR + "auto-mpg.csv", "auto-mpg", null, -1)
  auto_tab.show()

  banner("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD(1 to 6, 0)
  println(s"x = $x")
  println(s"y = $y")

  //  val (x, y) = pullResponse (xy)


  banner("regression")
  val ox = VectorD.one (x.dim1) +^: x
  val rg = new Regression(x, y)
  println(rg.analyze().report)
  println(rg.summary)
}

object Regressionforward extends App
{
  import scalation.columnar_db.Relation
//
  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto-mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

//  val (x, y) = pullResponse (xy)


  banner ("auto_mpg regression")
  val rg = new Regression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)

  banner ("forward regression")
  val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  println (s"k = $k, n = ${x.dim2}")
  val t = VectorD.range (1, rSq.dim1)                            // instance index
  //    val t = VectorD.range (1, k)                                   // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for Regression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Regressionforward", lines = true).saveImage("R^2 vs n for Regressionforward")


  println (s"rSq = $rSq")
  println (s"cols = $cols")

} // Regressionforward object

object Regressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")


  banner ("auto_mpg regression")
  val rg = new Regression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)

  banner ("backward regression")
  val (cols, rSq) = rg.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  println (s"k = $k, n = ${x.dim2}")
  val t = VectorD.range (1, rSq.dim1)                            // instance index
  //    val t = VectorD.range (1, k)                                   // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for Regression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Regressionbackward", lines = true).saveImage("R^2 vs n for Regressionbackward")


  println (s"rSq = $rSq")
  println (s"cols = $cols")

} // Regressionbackward object

object Regressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")


  banner ("auto_mpg regression")
  val rg = new Regression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)


  banner ("stepwise regression")
  val (cols, rSq) = rg.stepwiseSelAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  println (s"k = $k, n = ${x.dim2}")
  val t = VectorD.range (0, rSq.dim1)                            // instance index
  //    val t = VectorD.range (1, k)                                   // instance index
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for Regressionstepwise", lines = true).saveImage("R^2 vs n for Regressionstepwise")

  println (s"rSq = $rSq")
  println (s"cols = $cols")

} // Regressionstepwise object
object QuadRegressionForward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg: QuadRegression")
  val qrg = new QuadRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                  // number of variables
  val nt = QuadRegression.numTerms (n)                             // number of terms
  println (qrg.summary)
  println (s"n = $n, nt = $nt")

  banner ("Forward Selection Test")
  val (cols, rSq) = qrg.forwardSelAll ()                           // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                      // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for QuadRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadRegressionForward", lines = true).saveImage("R^2 vs n for QuadRegressionForward")

  println (s"rSq = $rSq")

} // QuadRegressionForward object

object QuadRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg: QuadRegression")
  val qrg = new QuadRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                  // number of variables
  val nt = QuadRegression.numTerms (n)                             // number of terms
  println (qrg.summary)
  println (s"n = $n, nt = $nt")

  banner ("backward elimination Test")
  val (cols, rSq) = qrg.backwardElimAll ()                           // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                    // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for QuadRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadRegressionbackward", lines = true).saveImage("R^2 vs n for QuadRegressionbackward")

  println (s"rSq = $rSq")

} // QuadRegressionbackward object

object QuadRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg: QuadRegression")
  val qrg = new QuadRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                  // number of variables
  val nt = QuadRegression.numTerms (n)                             // number of terms
  println (qrg.summary)
  println (s"n = $n, nt = $nt")

  banner ("stepwise elimination Test")
  val (cols, rSq) = qrg.stepwiseSelAll ()                           // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (0, rSq.dim1)                                    // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for QuadRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadRegressionstepwise", lines = true).saveImage("R^2 vs n for QuadRegressionstepwise")

  println (s"rSq = $rSq")

} // QuadRegressionstepwise object

object QuadXRegressionForward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val qrg = new QuadXRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                // number of variables
  val nt = QuadXRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (qrg.summary)

  banner ("Forward Selection Test")
  val (cols, rSq) = qrg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                  // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for QuadXRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadXRegressionForward", lines = true).saveImage("R^2 vs n for QuadXRegressionForward")

  println (s"rSq = $rSq")

} // QuadXRegressionForward object

object QuadXRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val qrg = new QuadXRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                // number of variables
  val nt = QuadXRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (qrg.summary)

  banner ("backward Selection Test")
  val (cols, rSq) = qrg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                   // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for QuadXRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadXRegressionbackward", lines = true).saveImage("R^2 vs n for QuadXRegressionbackward")

  println (s"rSq = $rSq")

} // QuadXRegressionbackward object

object QuadXRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val qrg = new QuadXRegression (x, y)
  println (qrg.analyze ().report)
  val n  = x.dim2                                                // number of variables
  val nt = QuadXRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (qrg.summary)

  banner ("stepwise Selection Test")
  val (cols, rSq) = qrg.stepwiseSelAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (0, rSq.dim1)                                   // instance index
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for QuadXRegressionstepwise", lines = true).saveImage("R^2 vs n for QuadXRegressionstepwise")

  println (s"rSq = $rSq")

} // QuadXRegressionstepwise object

object CubicRegressionforward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)

  //  import ExampleAutoMPG.{x, y}

  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("Forward Selection Test")
  val (cols, rSq) = crg.forwardSelAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size - 1
  val t = VectorD.range (1, rSq.dim1)                                  // instance index
//  new PlotM (t, rSq.t, lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicRegressionforward", lines = true).saveImage("R^2vsnforCubicRegressionforward")

  println (s"k = $k, nt = $nt")
  println (s"rSq = $rSq")

} // CubicRegressionforward object

object CubicRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)

  //  import ExampleAutoMPG.{x, y}

  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("backward Selection Test")
  val (cols, rSq) = crg.backwardElimAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size - 1
  val t = VectorD.range (1, rSq.dim1)                                 // instance index
//  new PlotM (t, rSq.t, lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicRegressionbackward", lines = true).saveImage("R^2vsnforCubicRegressionbackward")

  println (s"k = $k, nt = $nt")
  println (s"rSq = $rSq")

} // CubicRegressionbackward object

object CubicRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)

  //  import ExampleAutoMPG.{x, y}

  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicRegression.numTerms (n)                          // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("stepwise Selection Test")
  val (cols, rSq) = crg.stepwiseSelAll ()                         // R^2, R^2 bar, R^2 cv
  val k = cols.size - 1
  val t = VectorD.range (0, rSq.dim1)                                  // instance index
//  new PlotM (t, rSq.t, lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicRegressionstepwise", lines = true).saveImage("R^2vsnforCubicRegressionstepwise")

  println (s"k = $k, nt = $nt")
  println (s"rSq = $rSq")

} // CubicRegressionstepwise object

object CubicXRegressionforward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicXRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicXRegression.numTerms (n)                         // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("Forward Selection Test")
  val (cols, rSq) = crg.forwardSelAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                // instance index
  println (s"t.dim = ${t.dim}, rSq.dim1 = ${rSq.dim1}")
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for CubicXRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicXRegressionforward", lines = true).saveImage("R^2vsnforCubicXRegressionforward")
  println (s"rSq = $rSq")

} // CubicXRegressionforward object

object CubicXRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicXRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicXRegression.numTerms (n)                         // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("Backward Selection Test")
  val (cols, rSq) = crg.backwardElimAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                  // instance index
  println (s"t.dim = ${t.dim}, rSq.dim1 = ${rSq.dim1}")
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for CubicXRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicXRegressionbackward", lines = true).saveImage("R^2vsnforCubicXRegressionbackward")

  println (s"rSq = $rSq")

} // CubicXRegressionbackward object

object CubicXRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val crg = new CubicXRegression (x, y)
  println (crg.analyze ().report)

  val n  = x.dim2                                                // number of variables
  val nt = CubicXRegression.numTerms (n)                         // number of terms
  println (s"n = $n, nt = $nt")
  println (crg.summary)

  banner ("stepwise Selection Test")
  val (cols, rSq) = crg.stepwiseSelAll ()                          // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (0, rSq.dim1)                                   // instance index
  println (s"t.dim = ${t.dim}, rSq.dim1 = ${rSq.dim1}")
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for CubicXRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for CubicXRegressionstepwise", lines = true).saveImage("R^2vsnforCubicXRegressionstepwise")
  println (s"rSq = $rSq")

} // CubicXRegressionstepwise object

object LassoRegressionforward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rg = new LassoRegression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)
  val n = x.dim2                                                    // number of parameters/variables

  banner ("Forward Selection Test")
  val (cols, rSq) = rg.forwardSelAll ()                          // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for LassoRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for LassoRegressionforward", lines = true).saveImage("R^2vsnforLassoRegressionforward")

  println (s"rSq = $rSq")

} // LassoRegressionforward object

object LassoRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rg = new LassoRegression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)
  val n = x.dim2                                                    // number of parameters/variables

  banner ("backward Selection Test")
  val (cols, rSq) = rg.backwardElimAll ()                          // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for LassoRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for LassoRegressionbackward", lines = true).saveImage("R^2vsnforLassoRegressionbackward")

  println (s"rSq = $rSq")

} // LassoRegressionbackward object

object LassoRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rg = new LassoRegression (x, y)
  println (rg.analyze ().report)
  println (rg.summary)
  val n = x.dim2                                                    // number of parameters/variables

  banner ("stepwise Selection Test")
  val (cols, rSq) = rg.stepwiseSelAll ()                          // R^2, R^2 Bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (0, rSq.dim1)                                  // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for LassoRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for LassoRegressionstepwise", lines = true).saveImage("R^2vsnforLassoRegressionstepwise")

  println (s"rSq = $rSq")

} // LassoRegressionstepwise object

object RidgeRegressionforward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
  println (rrg.analyze ().report)
  println (rrg.summary)
  val n = x.dim2                                                     // number of variables

  banner ("Forward Selection Test")
  val (cols, rSq) = rrg.forwardSelAll ()                             // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                   // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for RidgeRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for RidgeRegressionforward", lines = true).saveImage("R^2vsnforRidgeRegressionforward")

  println (s"rSq = $rSq")

} // RidgeRegressionforward object

object RidgeRegressionbackward extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
  println (rrg.analyze ().report)
  println (rrg.summary)
  val n = x.dim2                                                     // number of variables

  banner ("backward Selection Test")
  val (cols, rSq) = rrg.backwardElimAll ()                             // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (1, rSq.dim1)                                     // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for RidgeRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for RidgeRegressionbackward", lines = true).saveImage("RidgeRegressionbackward")

  println (s"rSq = $rSq")

} // RidgeRegressionbackward object

object RidgeRegressionstepwise extends App
{
  import scalation.columnar_db.Relation

  banner ("auto_mpg relation")
  val auto_tab = Relation (BASE_DIR + "auto-mpg.csv", "auto_mpg", null, -1)
  auto_tab.show ()

  banner ("auto_mpg (x, y) dataset")
  val (x, y) = auto_tab.toMatriDD (1 to 6, 0)
  println (s"x = $x")
  println (s"y = $y")

  banner ("auto_mpg regression")
  val rrg = RidgeRegression (x, y, null, RidgeRegression.hp, Cholesky)
  println (rrg.analyze ().report)
  println (rrg.summary)
  val n = x.dim2                                                     // number of variables

  banner ("Forward Selection Test")
  val (cols, rSq) = rrg.stepwiseSelAll ()                             // R^2, R^2 bar, R^2 cv
  val k = cols.size
  val t = VectorD.range (0, rSq.dim1)                                 // instance index
//  new PlotM (t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
//    "R^2 vs n for RidgeRegression", lines = true)
  new PlotM(t, rSq.t, Array ("R^2", "R^2 bar", "R^2 cv"),
    "R^2 vs n for RidgeRegressionstepwise", lines = true).saveImage("RidgeRegressionstepwise")

  //  New PlotM(“content here for plot”).saveImage(“name of the plot”)

  println (s"rSq = $rSq")

} // RidgeRegressionstepwise object