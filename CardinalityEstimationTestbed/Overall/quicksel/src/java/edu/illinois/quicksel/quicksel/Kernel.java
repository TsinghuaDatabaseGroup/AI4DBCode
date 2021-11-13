package edu.illinois.quicksel.quicksel;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import org.apache.commons.lang3.tuple.Pair;
import org.jblas.DoubleMatrix;

public class Kernel extends Hyperrectangle {

  static int split_factor = 2;
  
  static Random rand = new Random();

  public Kernel(Hyperrectangle rec) {
    super(rec);
  }

  public Kernel(List<Pair<Double, Double>> intervals) {
    super(intervals);
  }
  
  public List<List<Double>> getCorners() {
    List<List<Double>> corners = new ArrayList<>();
    corners.add(new ArrayList<Double>());
    
    // in each iteration, increase the dimension of the corners by one
    for (Pair<Double, Double> interval : intervals) {
      List<List<Double>> augmentedCorners = new ArrayList<>();
      for (List<Double> c : corners) {
        List<Double> c1 = new ArrayList<>(c);   // copy before augment
        c1.add(interval.getLeft());
        augmentedCorners.add(c1);
        
        List<Double> c2 = new ArrayList<>(c);   // copy before augment
        c2.add(interval.getRight());
        augmentedCorners.add(c2);
//        c.add(interval.getLeft());
//        c.add(interval.getRight());
      }
      corners = augmentedCorners;
    }
    
    assert(corners.size() == Math.pow(2, d));
    
    // remove the corners on the min or max points
//    List<List<Double>> reducedCorners = new ArrayList<>();
//    for (List<Double> c : corners) {
//      boolean onEdge = false;
//      for (double d : c) {
//        if (d <= 1e-6 || d >= 1.0-1e-6) {
//          onEdge = true;
//        }
//      }
//      if (!onEdge) {
//        reducedCorners.add(c);
//      }
//    }
    
    return corners;
  }
  
  public List<Double> getRandomCenter() {
    List<Double> center = new ArrayList<>();
    for (Pair<Double, Double> interval : intervals) {
      double r = rand.nextDouble();
      center.add(interval.getLeft() + r * (interval.getRight() - interval.getLeft()));
    }
    return center;
  }
  
  public List<Double> getCenter() {
    List<Double> center = new ArrayList<>();
    for (Pair<Double, Double> interval : intervals) {
      center.add((interval.getLeft() + interval.getRight()) / 2.0);
    }
    return center;
  }

  // one to one correspondence without any split
  public Kernel(Query query, Hyperrectangle min_max) {
    super(query.rectangleFromQuery(min_max));
  }

  public static List<Kernel> splitQueryToMultipleKernels(Query query, Hyperrectangle min_max) {
    return splitQueryToMultipleKernels(query, min_max, Kernel.split_factor);
  }

  /**
   * Generates multiple kernels by dividing the hyperrectangle for 'query'.
   * Each dimension specified in the 'query' is split into split_factor number of
   * intervals.
   * @param query
   * @param split_factor
   * @return
   */
  public static List<Kernel> splitQueryToMultipleKernels(Query query, Hyperrectangle min_max, int split_factor) {
    List<Kernel> kernels = new ArrayList<Kernel>();

    int qd = query.getConstraints().keySet().size();

    for (int idx = 0; idx < Math.pow(split_factor, qd); idx++) {
      Query q1 = new Query(query);

      int j = 0;
      Iterator<Entry<Integer, Pair<Double, Double>>> it = query.getConstraints().entrySet().iterator();
      while (it.hasNext()) {
        Map.Entry<Integer, Pair<Double, Double>> pair = (Map.Entry<Integer, Pair<Double, Double>>) it.next();
        int jb = (idx / ((int) Math.round(Math.pow(split_factor, j)))) % split_factor;
        Pair<Double, Double> intvl = pair.getValue();

        double size = intvl.getRight() - intvl.getLeft();
        Pair<Double, Double> subrange = Pair.of(
            intvl.getLeft() + size / ((double) split_factor) * jb,
            intvl.getLeft() + size / ((double) split_factor) * (jb+1) );
        q1.getConstraints().put(pair.getKey(), subrange);

        j += 1;
      }

      kernels.add(new Kernel(q1, min_max));
    }

    return kernels;
  }
  
  @Override
  public String toString() {
    StringBuilder builder = new StringBuilder();
    for (Pair<Double, Double> interval : intervals) {
      builder.append(String.format(" %.3f", interval.getLeft()));
      builder.append(String.format(" %.3f", interval.getRight()));
    }
    return builder.toString();
  }

  /**
   * Creates two (m,k)-dim matrices.
   * k is the dimension of the domain.
   * m is the number of kernels.
   * The first matrix is the starting points of ranges.
   * The second matrix is the end points of ranges. 
   * @param kernels
   * @return
   */
  public static Pair<DoubleMatrix, DoubleMatrix> kernelsToMat(List<? extends Hyperrectangle> kernels) {	
    return Hyperrectangle.recsToMat(kernels);
  }
}