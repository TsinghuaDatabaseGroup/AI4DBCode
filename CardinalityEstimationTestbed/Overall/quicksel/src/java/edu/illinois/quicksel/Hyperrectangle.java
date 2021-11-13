package edu.illinois.quicksel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

import static java.lang.Math.*;


/**
 * Literally, represents multidimensional rectangle.
 * 
 * @author Yongjoo Park
 */
public class Hyperrectangle {

  /**
   * dimension
   */
  public int d;
  
  /**
   * Range for each dimension; the size must be 'd'.
   */
  public List<Pair<Double, Double>> intervals = new ArrayList<Pair<Double, Double>>();

  public Hyperrectangle(List<Pair<Double, Double>> intervals) {
    this.d = intervals.size();

    for (Pair<Double, Double> p : intervals) {
      assert(p.getLeft() < p.getRight());
      this.intervals.add(Pair.of(new Double(p.getLeft()), new Double(p.getRight())));
    }
  }

  // copy constructor
  public Hyperrectangle(Hyperrectangle other) {
    this(other.intervals);
  }

  public String toString() {
    return intervals.toString();
  }

  public Query toQuery() {
    HashMap<Integer, Pair<Double, Double>> m = new HashMap<Integer, Pair<Double, Double>>();

    for (int i = 0; i < this.d; i++) {
      m.put(i, intervals.get(i));
    }

    return new Query(m);
  }

  public double vol() {
    return this.intersect(this);
  }

  public double intersect(Hyperrectangle other) {
    double vol = 1.0;

    for (int i = 0; i < this.d; i++) {
      Pair<Double, Double> p = this.intervals.get(i);
      double a = p.getLeft();
      double b = p.getRight();

      Pair<Double, Double> o = other.intervals.get(i);
      double c = o.getLeft();
      double d = o.getRight();

      vol = vol * computeInterval(a, b, c, d);
    }

    return vol;
  }

  private double computeInterval(double a, double b, double c, double d) {
    assert(a < b); assert(c < d);
    if (b <= c) return 0;
    if (d <= a) return 0;
    return min(b, d) - max(a, c);
  }

  public double intersectProportion(Hyperrectangle other) {
    return this.intersect(other) / this.vol();
  }

  /**
   * Creates two (m,k)-dim matrices.
   * k is the dimension of the domain.
   * m is the number of rectangles.
   * The first matrix is the starting points of ranges.
   * The second matrix is the end points of ranges. 
   * @param recs
   * @return
   */
  public static Pair<DoubleMatrix, DoubleMatrix> recsToMat(List<? extends Hyperrectangle> recs) {
    assert(recs.size() > 0);
    int m = recs.size();
    int k = recs.get(0).d;

    DoubleMatrix first = DoubleMatrix.zeros(m, k);
    DoubleMatrix second = DoubleMatrix.zeros(m, k);

    for (int i = 0; i < m; i++) {
      Hyperrectangle rec = recs.get(i);

      for (int j = 0; j < k; j++) {
        first.put(i, j, rec.intervals.get(j).getLeft());
        second.put(i, j, rec.intervals.get(j).getRight());
      }
    }

    return Pair.of(first, second);
  }
}
