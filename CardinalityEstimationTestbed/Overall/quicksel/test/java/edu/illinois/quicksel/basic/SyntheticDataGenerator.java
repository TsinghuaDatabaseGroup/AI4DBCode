package edu.illinois.quicksel.basic;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;

import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Query;
import org.apache.commons.math3.distribution.NormalDistribution;


public class SyntheticDataGenerator {

  public static List<List<Double>> twoDimGaussianDataset(int size) {

    List<List<Double>> dataset = new ArrayList<>();

    MultivariateNormalDistribution normal = new MultivariateNormalDistribution(
        new double[]{0.5, 0.5},
        new double[][]{new double[]{0.09, 0.04}, new double[]{0.04, 0.09}});

    for (int i = 0; i < size; i++) {	
      dataset.add(new ArrayList<Double>(Arrays.asList(ArrayUtils.toObject(normal.sample()))));
    }

    return dataset;
  }

  public static List<Assertion> workflowAssertion(List<List<Double>> dataset) {
    /**
     * cc = linspace(0.1, 0.9, 1000)
     * For c in cc:
     *   # cr is the center of 2D rectangle.
     *   # randn() generaets a random variables from a standard normal
     *   cr = [c+randn()*0.1 c+randn()*0.1]
     *   create a rectangle R whose width is 0.2 and height 0.2; its center is cr.
     *   new_assertion = (R, selectivity(R))
     */
    Vector<Assertion> queryset = new Vector<Assertion>();
    List<Double> cc = new ArrayList<>();
    for (double i=0.1;i<=0.9;i+=0.0008) {
      cc.add(i);
    }
    Random rand = new Random();
    for (double c:cc) {
      double r = rand.nextDouble();
      Pair<Double, Double> cr = new ImmutablePair<>(c-r*0.1, c+r*0.1);
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      Double x1 = (cr.getLeft() - 0.1)<0? 0:cr.getLeft()-0.1;
      Double x2 = (cr.getLeft() + 0.1)>=1? 1:cr.getLeft()+0.1;
      Double y1 = (cr.getRight() - 0.1)<0? 0:cr.getRight()-0.1;
      Double y2 = (cr.getRight() + 0.1)>=1? 1:cr.getRight()+0.1;
      r1.put(0, Pair.of(x1, x2));
      r1.put(1, Pair.of(y1, y2));
      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));

    }

    return queryset;

  }

  public static List<List<Double>> twoDimGaussianDataset(double correlation, int size) {

    List<List<Double>> dataset = new ArrayList<>();
//    double[] standardDeviation = new double[]{1.0, 1.0};
//    double[][] correlationMatrix = new double[][]{new double[]{1.0, correlation}, new double[]{correlation, 1.0}};
    // Coveriance Matrix = Diag(std)*correlation*Diag(std), so in this case, correlationMatrix=covarianceMatrix
    double stddev = 0.3;
    double cov = correlation * stddev * stddev;
    double var = stddev * stddev;
    double[][] covarianceMatrix = new double[][] {new double[] {var, cov}, new double[] {cov, var}};
    
    MultivariateNormalDistribution normal = new MultivariateNormalDistribution(
        new double[]{0.5, 0.5},
        covarianceMatrix);

    for (int i = 0; i < size; i++) {
      dataset.add(new ArrayList<Double>(Arrays.asList(ArrayUtils.toObject(normal.sample()))));
    }

    return dataset;
  }

  public static List<List<Double>> nDimGaussianDataset(int n, int size) {

    List<List<Double>> dataset = new ArrayList<>();

    double stddev = 0.2; //variance =0.04
    double mean=0.5;
    NormalDistribution normal = new NormalDistribution(mean, stddev);

    for (int i = 0; i < size; i++) {
      // values in each dimension are generated independently
      List<Double> data = new ArrayList<>();
      for (int j=0;j<n;j++) {
        data.add(normal.sample());
      }
      dataset.add(data);
    }

    return dataset;
  }
  
  // queries highly overlap
  public static List<Assertion> twoDimGaussianDatasetRandomAssertions(int size, List<List<Double>> dataset) {

    Vector<Assertion> queryset = new Vector<Assertion>();
    Random rand = new Random(0);

    for (int i = 0; i < size; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      r1.put(0, Pair.of(rand.nextDouble() * 0.5, rand.nextDouble() * 0.5 + 0.5));
      r1.put(1, Pair.of(rand.nextDouble() * 0.5, rand.nextDouble() * 0.5 + 0.5));
      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }
    return queryset;
  }

  public static List<Assertion> nDimRandomAssertions(int size, List<List<Double>> dataset) {
    Vector<Assertion> queryset = new Vector<Assertion>();
    int dim = dataset.get(0).size();
    Random rand = new Random(0);

    for (int i = 0; i < size; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      // for each dim, the interval size is (0.1)^(1/d)
      for (int j=0;j<dim;j++) {
        double left = rand.nextDouble() * 0.5;
        double right = left + 0.5;
        r1.put(j, Pair.of(left, right));
      }

      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }
    return queryset;
  }

  public static List<Assertion> nDimRandomAssertionsSelectivityFixed(int size, List<List<Double>> dataset) {
    Vector<Assertion> queryset = new Vector<Assertion>();
    int dim = dataset.get(0).size();
    Random rand = new Random(0);

    for (int i = 0; i < size; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      // for each dim, the interval size is (0.1)^(1/d)
      for (int j=0;j<dim;j++) {
        double left = -1;
        while (left<0) {
          left = rand.nextDouble()*(1-Math.pow(0.1, 1.0/(double) dim));
        }
        double right = left+Math.pow(0.1, 1.0/(double) dim);
        r1.put(j, Pair.of(left, right));
      }

      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }
    return queryset;
  }
  
  // queries less overlap
  public static List<Assertion> twoDimGaussianDatasetRandomAssertions2(int size, List<List<Double>> dataset) {

    Vector<Assertion> queryset = new Vector<Assertion>();
    Random rand = new Random(0);

    for (int i = 0; i < size; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      double d1 = rand.nextDouble()*0.9;
      double d2 = rand.nextDouble()*0.9;
      r1.put(0, Pair.of(d1, d1+0.1));
      r1.put(1, Pair.of(d2, d2+0.1));
      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }

    return queryset;
  }

  public static List<Query> twoDimGaussianDatasetRandomTestQueries(int size) {

    Vector<Query> queryset = new Vector<Query>();
    Random rand = new Random(1);

    for (int i = 0; i < size; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      r1.put(0, Pair.of(rand.nextDouble() * 0.5, rand.nextDouble() * 0.5 + 0.5));
      r1.put(1, Pair.of(rand.nextDouble() * 0.5, rand.nextDouble() * 0.5 + 0.5));
      queryset.add(new Query(r1));
    }

    return queryset;
  }
  
  public static List<Assertion> twoDimDefaultAssertion(List<List<Double>> dataset) {
    List<Assertion> assertions = new ArrayList<Assertion>();
    
    HashMap<Integer, Pair<Double, Double>> constraints = new HashMap<>();
    constraints.put(0, Pair.of(0.0, 1.0));
    Query q = new Query(constraints);
    assertions.add(new Assertion(q, 1.0));
    
    return assertions;
  }

  public static List<Assertion> nDimDefaultAssertion(int n, List<List<Double>> dataset) {
    List<Assertion> assertions = new ArrayList<Assertion>();

    HashMap<Integer, Pair<Double, Double>> constraints = new HashMap<>();
    for (int i=0;i<n;i++) {
      constraints.put(i, Pair.of(0.0, 1.0));
    }
    Query q = new Query(constraints);
    assertions.add(new Assertion(q, 1.0));

    return assertions;
  }

  public static List<Assertion> twoDimPermanentAssertions(int gridNum, List<List<Double>> dataset) {

    List<Assertion> queryset = new Vector<Assertion>();

    double step = 1.0 / ((double) gridNum);

    for (int i = 0; i < gridNum; i++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      r1.put(0, Pair.of(step*i, step*(i+1)));
      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }

    for (int j = 0; j < gridNum; j++) {
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
      r1.put(1, Pair.of(step*j, step*(j+1)));
      Query q1 = new Query(r1);
      queryset.add(new Assertion(q1, countSatisfyingItems(dataset, q1) / ((double) dataset.size())));
    }

    return queryset;
  }

  public static List<List<Query>> twoDimGaussianDatasetGridTestQueries(int gridNum) {

    List<List<Query>> queryset = new ArrayList<>();

    double step = 1.0 / ((double) gridNum);

    for (int i = 0; i < gridNum; i++) {
      Vector<Query> vq = new Vector<Query>();
      for (int j = 0; j < gridNum; j++) {
        HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
        r1.put(0, Pair.of(step*i, step*(i+1)));
        r1.put(1, Pair.of(step*j, step*(j+1)));
        vq.add(new Query(r1));
      }
      queryset.add(vq);
    }

    return queryset;
  }

  public static int countSatisfyingItems(List<List<Double>> dataset, Query query) {
    int count = 0;

    for (int i = 0; i < dataset.size(); i++) {
      if (query.doesSatisfy(dataset.get(i))) {
        count += 1;
      }
    }

    return count;
  }

}
