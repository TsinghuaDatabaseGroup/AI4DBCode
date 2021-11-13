package edu.illinois.quicksel.quicksel;

import static java.lang.Math.max;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.Vector;

import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.OptimizerWrapper;
import edu.illinois.quicksel.Query;
import org.jblas.DoubleMatrix;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.SelectivityLearning;


/**
 * Selectivity learning method based on uniform mixture model.
 * 
 * This method is our technical contribution.
 * @author Yongjoo Park
 */
public class QuickSel implements SelectivityLearning {

  private Hyperrectangle min_max;

  // these assertions are not removed.
  private List<Assertion> permanent_assertions = new ArrayList<Assertion>();

  // these assertions may be removed by 'reduceNumberOfAssertions'
  private List<Assertion> assertions = new ArrayList<Assertion>();

  private List<Kernel> kernels;

  private List<Double> weights;

  private Pair<DoubleMatrix, DoubleMatrix> kernelMat;

  private DoubleMatrix normalizedWeightsMat;

  final private int num_var_limit = 100000;
  
  private int enforced_num_var = -1;   // used only positive.
  
  private final double kernel_scale_factor = 1.0;

  private DoubleMatrix A;
  
  private DoubleMatrix P;
  
  private DoubleMatrix vv;
  
  private String subpopulationMode = "sampling";  // or "kmeans"
  
  // each element is the per-dim distance.
  private List<List<Double>> distancesToFarthestPoint;

  public QuickSel(Hyperrectangle min_max_range) {
    this(min_max_range, 1.0);
  }

  public QuickSel(Hyperrectangle min_max_range, double total_freq) {
    min_max = min_max_range;
  }
  
  public void setSubpopulationModel(String mode) {
    subpopulationMode = mode;
  }
  
  public int getEnforcedVarCount() {
    return enforced_num_var;
  }
  
  public void setEnforcedVarCount(int count) {
    enforced_num_var = count;
  }

  @Override
  public String toString() {
    String ret = "";
    assert(kernels.size() == weights.size());

    for (int i = 0; i < kernels.size(); i++) {
      ret += kernels.get(i) + "\t" + weights.get(i).toString();
      ret += "\n";
    }

    return ret;
  }

  @Override
  public void addAssertion(Assertion a) {
    assertions.add(a);	
  }

  @Override
  public void addPermanentAssertion(Assertion a) {
    permanent_assertions.add(a);	
  }

  @Override
  public void assignOptimalWeights() {
    assignOptimalWeights(false, 1e6);
  }

  private static final double constraint_weight = 1e6;

  public void assignOptimalWeights(boolean debug_output) {
    assignOptimalWeights(debug_output, constraint_weight);
  }

  public int getAdaptiveVarCountLimit() {
    if (enforced_num_var > 0) {
      return enforced_num_var;
    }
    
    int assertionCount = assertions.size();
    return Math.min(assertionCount*4, num_var_limit);
  }

  private List<Kernel> regenerateKernels(List<Kernel> originalKernels) {

    List<List<Double>> centers = new ArrayList<>();
    for (Kernel k : originalKernels) {
      //      centers.add(k.getCenter());
      //      centers.add(k.getRandomCenter());
      centers.addAll(k.getCorners());
    }

    // if the problem size is too small, we add some random centers
//    while (centers.size() < getAdaptiveVarCountLimit() * 5) {
    while (centers.size() < getAdaptiveVarCountLimit() * 5) {
      for (Kernel k : originalKernels) {
        centers.add(k.getRandomCenter());
      }
    }
    
//    printPoints(centers);
    
    System.out.println("Limit points: " + new Timestamp(System.currentTimeMillis()));

    int eventualCenterCount = Math.min(centers.size(), getAdaptiveVarCountLimit());
    centers = removeDuplicates(centers);
    
    List<Kernel> newKernels = null;
    
    if (subpopulationMode.equals("sampling")) {
      centers = limitCenterCount(centers, eventualCenterCount);
  //    centers = removeDuplicates(centers);
  //    printPoints(centers);
      // generate kernels from the centers
      System.out.println("Generates new kernels: " + new Timestamp(System.currentTimeMillis()));
      newKernels = generateKernelsFromCenters(centers);
//      List<Kernel> newKernels = generateKernelsFromCenters2(centers);
    }
    else if (subpopulationMode.equals("kmeans")) {
      System.out.println("Before limiting: " + centers.size());
      centers = limitCenterCountForKmeans(centers, eventualCenterCount);
      System.out.println("After limiting: " + centers.size());
      newKernels = generateKernelsUsingKmeans(centers, eventualCenterCount);
//      for (Kernel k : newKernels) {
//        System.out.println(k);
//      }
    }
    else {
      throw new RuntimeException("unexpected");
    }
    
    System.out.println("Done generating kernels: " + new Timestamp(System.currentTimeMillis()));
    
    return newKernels;
  }
  
  private List<List<Double>> limitCenterCountForKmeans(
      List<List<Double>> centers, 
      int clusterCount) 
  {
    
    if (centers.size() < clusterCount * 10) {
      return centers;
    }
    
    List<List<Double>> newCenters = new ArrayList<>();
    Collections.shuffle(centers);
    newCenters.addAll(centers.subList(0, clusterCount * 10));
    return newCenters;
  }
  
  
  private List<Kernel> generateKernelsUsingKmeans(
      List<List<Double>> centers,
      int eventualCenterCount) 
  {
    List<Kernel> kernels = new ArrayList<Kernel>();
//    List<List<Double>> newCenters = new ArrayList<List<Double>>();
    
    List<Point> points = new ArrayList<>();
    for (List<Double> c : centers) {
      points.add(new Point(c));
    }

    int max_iter = 0;
    KMeansPlusPlusClusterer<Point> clusterer = 
        new KMeansPlusPlusClusterer<>(eventualCenterCount, max_iter);
    
    System.out.println("Started K-means++: " + eventualCenterCount + " out of " + centers.size());
    List<CentroidCluster<Point>> centroids = clusterer.cluster(points);
    System.out.println("Finished K-means++");

    // convert them to kernels
    Set<List<Double>> appearedCenter = new HashSet<>();
    for (CentroidCluster<Point> c : centroids) {
      double[] centerPoint = c.getCenter().getPoint();
      List<Double> centerObject = Arrays.asList(ArrayUtils.toObject(centerPoint));
      if (appearedCenter.contains(centerObject)) {
        continue;
      }
      appearedCenter.add(centerObject);
      
      List<Point> clusterPoints = c.getPoints();
      
      double[] min_corner = centerPoint.clone();   // bottom left corner
      double[] max_corner = centerPoint.clone();   // top right corner
      for (Point point : clusterPoints) {
        double[] p = point.getPoint();
        
        for (int i = 0; i < p.length; i++) {
          if (p[i] < min_corner[i]) {
            min_corner[i] = p[i];
          }
          if (p[i] > max_corner[i]) {
            max_corner[i] = p[i];
          }
        }
      }
      
      // sanity check
      boolean singular = false;
      for (int i = 0; i < centerPoint.length; i++) {
        if (min_corner[i] == max_corner[i]) {
          singular = true;
        }
      }
      
      if (singular) {
        continue;
      }
      
      // create a new kernel
      final double scale = 5.0;
      List<Pair<Double, Double>> intervals = new ArrayList<>();
      for (int i = 0; i < centerPoint.length; i++) {
        intervals.add(Pair.of(
            centerPoint[i] - (centerPoint[i] - min_corner[i]) * scale, 
            centerPoint[i] + (max_corner[i] - centerPoint[i]) * scale 
            ));
      }
      kernels.add(new Kernel(intervals));
    }
    
    System.out.println("Number of generated kernels: " + kernels.size());
    return kernels;
  }

  private void printPoints(List<List<Double>> points) {
    System.out.println("Points:");
    for (List<Double> p : points) {
      for (double d : p) {
        System.out.print(" " + d);
      }
      System.out.println();
    }
    System.out.println();
  }

  private List<List<Double>> removeDuplicates(List<List<Double>> centers) {
    Set<List<Double>> set = new HashSet<>();
    for (List<Double> c : centers) {
      set.add(c);
    }
    return new ArrayList<List<Double>>(set);
  }

  private List<List<Double>> limitCenterCount(List<List<Double>> centers, int clusterCount) {
    if (centers.size() <= getAdaptiveVarCountLimit()) {
      return centers;
    }

    List<List<Double>> newCenters = new ArrayList<>();
    
//    String method = "kmeans";
    String method = "sampling";
    
    if (method.equals("kmeans")) {
      // if the number of centers is larger than the predefined limit, reduce them with kmeans.
      // convert the centers to the compatible interface
      List<Point> points = new ArrayList<>();
      for (List<Double> c : centers) {
        points.add(new Point(c));
      }
  
      int max_iter = 0;
      KMeansPlusPlusClusterer<Point> clusterer = new KMeansPlusPlusClusterer<>(clusterCount, max_iter);
      List<CentroidCluster<Point>> centroids = clusterer.cluster(points);
  
      // convert back to our interface
      for (CentroidCluster<Point> c : centroids) {
        double[] point = c.getCenter().getPoint();
        List<Double> dp = new ArrayList<>();
        for (int i = 0; i < point.length; i++) {
          dp.add(point[i]);
        }
        newCenters.add(dp);
      }
    } else if (method.equals("sampling")) {
      Collections.shuffle(centers);
      newCenters.addAll(centers.subList(0, clusterCount));
    }
    
//    // computes the average distance to the clustered points
//    distancesToFarthestPoint = new ArrayList<>();
//    for (CentroidCluster<Point> c : centroids) {
//      List<Point> clusterPoints = c.getPoints();
//      Clusterable center = c.getCenter();
//      
//      List<Double> maxPerDimDistance = new ArrayList<>();
//      for (int i = 0; i < center.getPoint().length; i++) {
//        maxPerDimDistance.add(0.0);
//      }
//      for (Point p : clusterPoints) {
//        List<Double> perdim = p.perDimDistance(center);
//        for (int i = 0; i < perdim.size(); i++) {
//          maxPerDimDistance.set(i, Math.max(maxPerDimDistance.get(i), perdim.get(i)));
//        }
//      }
//      
//      distancesToFarthestPoint.add(maxPerDimDistance);
//    }

    return newCenters;
  }
  
  private double kernel_scale_factor2 = 3.0;
  
  private List<Kernel> generateKernelsFromCenters2(List<List<Double>> centers) {
    List<Kernel> newKernels = new ArrayList<>();
    int d = centers.get(0).size();
    
    for (int i = 0; i < centers.size(); i++) {
      List<Double> center = centers.get(i);
      List<Double> maxPerDimDist = distancesToFarthestPoint.get(i);
      List<Pair<Double, Double>> intervals = new ArrayList<>();
      for (int j = 0; j < d; j++) {
        intervals.add(Pair.of(
            center.get(j) - kernel_scale_factor2 * maxPerDimDist.get(j), 
            center.get(j) + kernel_scale_factor2 * maxPerDimDist.get(j)));
      }
      newKernels.add(new Kernel(intervals));
    }
    
    return newKernels;
  }

  private List<Kernel> generateKernelsFromCenters(List<List<Double>> centers) {
    List<Kernel> newKernels = new ArrayList<>();
    //    int dim = min_max.d;

    // we set the size of a kernel by inspecting the 'checkCount' closest kernels (based on their centers)
    int checkCount = 10;
    int checkSampleSize = Math.min(200, centers.size());
    
    List<List<Double>> copiedCenters = new ArrayList<>(centers);
    Collections.shuffle(centers);
    centers = centers.subList(0, checkSampleSize);
    for (List<Double> reference : copiedCenters) {
      // sort 
      Collections.sort(centers, new Comparator<List<Double>>() {
        @Override
        public int compare(List<Double> o1, List<Double> o2) {
          double d1 = distance(o1, reference);
          double d2 = distance(o2, reference);
          if (d1 < d2) {
            return -1;
          } else if (d1 > d2) {
            return 1;
          } else {
            return 0;
          }
        }
      });

      // compute average per-dim absolute displacement
      List<List<Double>> neighbors = centers.subList(1, Math.min(checkCount+1, centers.size()));
      List<List<Double>> disp = new ArrayList<>();
      for (List<Double> n : neighbors) {
        disp.add(absDisplacement(reference, n));
      }
      List<Double> perdim = perdimAverage(disp);

      // create a new kernel
      List<Pair<Double, Double>> intervals = new ArrayList<>();
      for (int i = 0; i < reference.size(); i++) {
        intervals.add(Pair.of(
            reference.get(i) - kernel_scale_factor * perdim.get(i),
            reference.get(i) + kernel_scale_factor * perdim.get(i)
            ));
      }
      newKernels.add(new Kernel(intervals));
    }

    return newKernels;
  }

  private List<Double> perdimAverage(List<List<Double>> distances) {
    List<Double> sumDist = new ArrayList<>();
    int d = distances.get(0).size();
    double k  = (double) distances.size();
    for (int i = 0; i < d; i++) {
      sumDist.add(1e-3);    // place a small default interval (to prevent 0)
    }

    // sum
    for (List<Double> dist : distances) {
      for (int i = 0; i < d; i++) {
        sumDist.set(i, sumDist.get(i) + dist.get(i));
      }
    }

    // divide by the number
    for (int i = 0; i < d; i++) {
      sumDist.set(i, sumDist.get(i) / k);
    }

    return sumDist;
  }

  private Double distance(List<Double> p1, List<Double> p2) {
    double dist = 0;
    for (int i = 0; i < p1.size(); i++) {
      dist += Math.pow(p1.get(i) - p2.get(i), 2);
    }
    return Math.sqrt(dist);
  }

  private List<Double> absDisplacement(List<Double> p1, List<Double> p2) {
    List<Double> dist = new ArrayList<>();
    for (int i = 0; i < p1.size(); i++) {
      dist.add(Math.abs(p1.get(i) - p2.get(i)));
    }
    return dist;
  }
  
  public void prepareOptimization() {
 // for permanent_assertions (including, min_max_range), only a single kernel is generated.
    // for other assertions (which include queries), we generate kernels depending on the attributes
    // each query touches.
    kernels = new ArrayList<Kernel>();
    for (int i = 0; i < permanent_assertions.size(); i++) {
      kernels.add(new Kernel(permanent_assertions.get(i).query.rectangleFromQuery(min_max)));
    }
    for (int i = 0; i< assertions.size(); i++) {
//            kernels.addAll(Kernel.splitQueryToMultipleKernels(assertions.get(i).query, min_max));
      kernels.add(new Kernel(assertions.get(i).query.rectangleFromQuery(min_max)));
    }
    
    System.out.println("Regenerates Kernels: " + new Timestamp(System.currentTimeMillis()));
    kernels = regenerateKernels(kernels);
    //    assert(newKernels.size() == kernels.size());

    // collect all assertions for convenience.
    List<Assertion> all_assertions = getAllAssertions();

    int n = all_assertions.size();
    int m = kernels.size();

    Pair<DoubleMatrix, DoubleMatrix> kmats = Kernel.kernelsToMat(kernels);
    this.kernelMat = kmats;    // store kernel matrix
    
    System.out.println("Computes intersections: " + new Timestamp(System.currentTimeMillis()));
    vv = intersections(kmats, kmats);
    System.out.println("Done computing intersections: " + new Timestamp(System.currentTimeMillis()));

    // intersections between every pair of kernels
    DoubleMatrix vvd = vv.diag().repmat(1, m);    // repmat(diag(vv), (m,1))
    P = vv.div(vvd).div(vvd.transpose());
    P.addi(DoubleMatrix.eye(m).muli(1e-3));
    //    System.out.println(P);

    // intersections between every pair of assertions and kernels
    List<Hyperrectangle> query_ranges = new ArrayList<Hyperrectangle>();
    for (Assertion a : all_assertions) {
      Hyperrectangle r = a.query.rectangleFromQuery(min_max);
      //      System.out.println(r);
      query_ranges.add(r);
    }

    Pair<DoubleMatrix, DoubleMatrix> qmats = Hyperrectangle.recsToMat(query_ranges);
    A = intersections(qmats, kmats);
    A = A.div(vv.diag().transpose().repmat(n, 1).add(1e-6));
  }
  
  private List<Assertion> getAllAssertions() {
    List<Assertion> all_assertions = new ArrayList<Assertion>();
    all_assertions.addAll(permanent_assertions);
    all_assertions.addAll(assertions);
    return all_assertions;
  }

  public void assignOptimalWeights(boolean debug_output, double constraints_weights) {
    List<Assertion> all_assertions = getAllAssertions();
    int n = all_assertions.size();
    double[] b = new double[n];
    for (int i = 0; i < n; i++) {
      b[i] = all_assertions.get(i).freq;
    }
    
    // Solves the problem
//    String method = "cvxopt";
    String method = "else";
    double[] sol;
    if (method.equals("cvxopt")) {
      // Solves the problem: cvxopt QP
      OptimizerWrapper.constraint_penalty = constraints_weights;
      sol = OptimizerWrapper.safe_squared_norm_solver(P, A, new DoubleMatrix(b));
    } else {
      // Solves the problem: our custom QP
      sol = ProjectionAlgorithm.solve(P, A, new DoubleMatrix(b), constraints_weights, 0);
    }

    if (debug_output) {
      DoubleMatrix x = new DoubleMatrix(sol);
//      System.out.println(String.format("A dim: (%d, %d)", A.rows, A.columns));
//      System.out.println(String.format("x dim: (%d, %d)", x.rows, x.columns));
      DoubleMatrix gap = A.mmul(x).sub(new DoubleMatrix(b));
      System.out.println("weights sum: " + x.sum());
      System.out.println("avg l1 gap: " + gap.norm1() / (double) gap.length);
      //      System.out.println(gap);
    }

    // set weights
    weights = new Vector<Double>();
    for (Double w : sol) {
      weights.add(w);
    }

    // store weights matrix for answering queries in the future.
    this.normalizedWeightsMat = (new DoubleMatrix(weights)).div(vv.diag());
  }

  /**
   * Fast computes the intersections between every row in 'aa' and every row in 'bb'.
   * @param aa
   * @param bb
   * @return (aa.rows, bb.rows)-dim matrix.
   */
  private DoubleMatrix intersections(Pair<DoubleMatrix, DoubleMatrix> aa, Pair<DoubleMatrix, DoubleMatrix> bb) {
    DoubleMatrix a = aa.getLeft();
    DoubleMatrix b = aa.getRight();
    DoubleMatrix c = bb.getLeft();
    DoubleMatrix d = bb.getRight();

    int n = a.rows;
    int m = c.rows;
    int dim = a.columns;

    DoubleMatrix intMat = DoubleMatrix.zeros(n, m);

    for (int j = 0; j < m; j++) {
      DoubleMatrix rc = c.getRow(j).repmat(n, 1);
      DoubleMatrix rd = d.getRow(j).repmat(n, 1);

      DoubleMatrix intervals = b.min(rd).sub(a.max(rc)).max(0);
      intMat.putColumn(j, intervals.getColumn(0));
      for (int k = 1; k < dim; k++) {
        intMat.putColumn(j, intMat.getColumn(j).mul(intervals.getColumn(k)));
      }
    }

    return intMat;
  }	

  @Override
  public void reduceNumberOfAssertions(int target_number) {
    target_number -= permanent_assertions.size();
    Vector<Assertion> reduced_assertions = new Vector<Assertion>();
    reduced_assertions.addAll(assertions.subList(max(0, assertions.size() - target_number), assertions.size()));
    assertions = reduced_assertions;
  }

  @Override
  public double answer(Query query) {
    List<Kernel> qq = new ArrayList<Kernel>();
    qq.add(new Kernel(query, min_max));
    Pair<DoubleMatrix, DoubleMatrix> qmat = Kernel.kernelsToMat(qq);

    return intersections(this.kernelMat, qmat).mul(normalizedWeightsMat).sum();
  }

  public int getVariableCount() {
    return kernels.size();
  }

  public void printKernels() {
    for (Kernel k : kernels) {
      System.out.println(k);
    }
  }
}
