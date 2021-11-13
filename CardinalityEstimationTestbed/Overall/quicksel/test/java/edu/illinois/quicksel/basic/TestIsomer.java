package edu.illinois.quicksel.basic;
import edu.illinois.quicksel.isomer.Isomer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.Assertion;


public class TestIsomer {

  public static void main(String[] args) {

    test2();
  }

  //	public static void test2() throws Exception {
  //	
  //		double[][] P = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
  ////		double[][] P = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
  //		double[] q = {0, 0, 0, 0};
  //		double[][] G = {{-1, 0, 0, 0}, {0, -1, 0, 0}, {0, 0, -1, 0}, {0, 0, 0, -1}};
  //		double[] h = {0, 0, 0, 0};
  //		double[][] A = {{1, 1, 1, 1}, {0, 1, 0, 1}, {0, 0, 1, 1}};
  //		double[] b = {1, 0.8, 0.3};
  //		double[] x = QPSolver.solve(P, q, G, h, A, b);
  //		
  //		Helpers.printVector(x);
  //	}

  public static void test1() {
    // TODO Auto-generated method stub

    //		System.out.println((new Isomer()).callme());

    Isomer isomer = defaultIsomer();

    // first query
    HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
    r1.put(0, Pair.of(1.0, 2.0));
    r1.put(1, Pair.of(0.0, 2.0));
    isomer.addAssertion(new Assertion(r1, 0.8));

    // second query
    HashMap<Integer, Pair<Double, Double>> r2 = new HashMap<Integer, Pair<Double, Double>>();
    r2.put(0, Pair.of(0.0, 2.0));
    r2.put(1, Pair.of(0.0, 1.0));
    isomer.addAssertion(new Assertion(r2, 0.3));

    isomer.assignOptimalWeights();

    //		System.out.println(isomer);
    //		
    //		isomer.reduceNumberOfNodes(2);

    System.out.println(isomer);

    // test query 1
    HashMap<Integer, Pair<Double, Double>> q1 = new HashMap<Integer, Pair<Double, Double>>();
    q1.put(0, Pair.of(0.0, 1.0));
    q1.put(1, Pair.of(0.0, 1.0));
    System.out.println("q1 answer:" + isomer.answer(new Query(q1)));

    // test query 2
    HashMap<Integer, Pair<Double, Double>> q2 = new HashMap<Integer, Pair<Double, Double>>();
    q2.put(0, Pair.of(1.0, 2.0));
    q2.put(1, Pair.of(0.0, 1.0));
    System.out.println("q2 answer:" + isomer.answer(new Query(q2)));

    // test query 3
    HashMap<Integer, Pair<Double, Double>> q3 = new HashMap<Integer, Pair<Double, Double>>();
    q3.put(0, Pair.of(0.0, 1.0));
    q3.put(1, Pair.of(1.0, 2.0));
    System.out.println("q3 answer:" + isomer.answer(new Query(q3)));

    // test query 4
    HashMap<Integer, Pair<Double, Double>> q4 = new HashMap<Integer, Pair<Double, Double>>();
    q4.put(0, Pair.of(1.0, 2.0));
    q4.put(1, Pair.of(1.0, 2.0));
    System.out.println("q4 answer:" + isomer.answer(new Query(q4)));
  }


  public static void test2() {
    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(100, dataset);

    System.out.println("dataset and query set generations done.");

    Isomer isomer = defaultIsomer(dataset);

    for (int i = 0; i < 10; i++) {
      isomer.addAssertion(assertions.get(i));
      System.out.println(String.format("(%d) node count: %d", i, isomer.countNodes()));
    }
    boolean debug_output = true;
    isomer.assignOptimalWeights(debug_output);
  }


  public static void test3() {

    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(2000, dataset);

    System.out.println("dataset and query set generations done.");

    Isomer isomer = defaultIsomer(dataset);

    for (int i = 0; i < 500; i++) {
      isomer.addAssertion(assertions.get(i));
      System.out.println(String.format("(%d) node count: %d", i, isomer.countNodes()));
    }
    isomer.assignOptimalWeights();

    long startTime = System.nanoTime();
    for (int i = 500; i < 1000; i++) {
      isomer.addAssertion(assertions.get(i));
      System.out.println(String.format("(%d) node count: %d", i, isomer.countNodes()));
    }
    isomer.reduceNumberOfAssertions(500);
    long elapsedTime1 = System.nanoTime() - startTime;
    isomer.assignOptimalWeights();
    long elapsedTime2 = System.nanoTime() - startTime;

    System.out.println(String.format("Elapsed time for the last 1,000 queries: %.3f sec and %.3f sec", elapsedTime1/1e9, elapsedTime2/1e9));

    //		System.out.println(isomer);
  }

  public static Isomer defaultIsomer() {
    List<Pair<Double, Double>> min_max = new ArrayList<>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = 1.0;
    int limit_node_count = 4000;
    return new Isomer(min_max_rec, total_freq, limit_node_count);
  }


  public static Isomer defaultIsomer(List<List<Double>> dataset) {
    List<Pair<Double, Double>> min_max = new ArrayList<>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = SyntheticDataGenerator.countSatisfyingItems(dataset, min_max_rec.toQuery()) / ((double) dataset.size());
    int limit_node_count = 4000;
    return new Isomer(min_max_rec, total_freq, limit_node_count);
  }

}
