package edu.illinois.quicksel.basic;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import edu.illinois.quicksel.quicksel.QuickSel;
import org.apache.commons.lang3.tuple.Pair;

import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.quicksel.Kernel;


public class TestQuickSel {

  public static void main(String[] args) {
    test1();
  }


  public static void query2Kernels() {
    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 2.0));
    min_max.add(Pair.of(0.0, 2.0));
    Hyperrectangle mix_max_range = new Hyperrectangle(min_max);
    QuickSel quickSel = new QuickSel(mix_max_range);

    HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
    r1.put(0, Pair.of(1.0, 2.0));
    Query q1 = new Query(r1);

    List<Kernel> kernels = Kernel.splitQueryToMultipleKernels(q1, mix_max_range);
    for (int i = 0; i < kernels.size(); i++) {
      System.out.println(kernels.get(i));
    }
  }


  public static void test1() {

    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 2.0));
    min_max.add(Pair.of(0.0, 2.0));
    QuickSel quickSel = new QuickSel(new Hyperrectangle(min_max));

    // first query
    HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
    r1.put(0, Pair.of(1.0, 2.0));
    quickSel.addAssertion(new Assertion(r1, 0.8));

    // second query
    HashMap<Integer, Pair<Double, Double>> r2 = new HashMap<Integer, Pair<Double, Double>>();
    r2.put(1, Pair.of(0.0, 1.0));
    quickSel.addAssertion(new Assertion(r2, 0.3));

    boolean debug_output = true;
    double constraints_weight = 1e6;
    quickSel.assignOptimalWeights(debug_output, constraints_weight);

    System.out.println(quickSel);

    // test query 1
    HashMap<Integer, Pair<Double, Double>> q1 = new HashMap<Integer, Pair<Double, Double>>();
    q1.put(0, Pair.of(0.0, 1.0));
    q1.put(1, Pair.of(0.0, 1.0));
    System.out.println("q1 answer:" + quickSel.answer(new Query(q1)));

    // test query 2
    HashMap<Integer, Pair<Double, Double>> q2 = new HashMap<Integer, Pair<Double, Double>>();
    q2.put(0, Pair.of(1.0, 2.0));
    q2.put(1, Pair.of(0.0, 1.0));
    System.out.println("q2 answer:" + quickSel.answer(new Query(q2)));

    // test query 3
    HashMap<Integer, Pair<Double, Double>> q3 = new HashMap<Integer, Pair<Double, Double>>();
    q3.put(0, Pair.of(0.0, 1.0));
    q3.put(1, Pair.of(1.0, 2.0));
    System.out.println("q3 answer:" + quickSel.answer(new Query(q3)));

    // test query 4
    HashMap<Integer, Pair<Double, Double>> q4 = new HashMap<Integer, Pair<Double, Double>>();
    q4.put(0, Pair.of(1.0, 2.0));
    q4.put(1, Pair.of(1.0, 2.0));
    System.out.println("q4 answer:" + quickSel.answer(new Query(q4)));
  }

  public static void test2() {

    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 2.0));
    min_max.add(Pair.of(0.0, 2.0));
    QuickSel quickSel = new QuickSel(new Hyperrectangle(min_max));

    // first query
    HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<Integer, Pair<Double, Double>>();
    r1.put(0, Pair.of(1.0, 2.0));
    r1.put(1, Pair.of(0.0, 2.0));
    quickSel.addAssertion(new Assertion(r1, 0.8));

    // second query
    HashMap<Integer, Pair<Double, Double>> r2 = new HashMap<Integer, Pair<Double, Double>>();
    r2.put(0, Pair.of(0.0, 2.0));
    r2.put(1, Pair.of(0.0, 1.0));
    quickSel.addAssertion(new Assertion(r2, 0.3));

    quickSel.assignOptimalWeights();

    System.out.println(quickSel);
  }


  public static void test3() {
    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(2000, dataset);

    System.out.println("dataset and query set generations done.");

    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 2.0));
    min_max.add(Pair.of(0.0, 2.0));
    QuickSel quickSel = new QuickSel(new Hyperrectangle(min_max));

    for (int i = 0; i < 500; i++) {
      quickSel.addAssertion(assertions.get(i));
    }

    boolean debug_output = true;
    double constraints_weight = 1e6;
    quickSel.assignOptimalWeights(debug_output, constraints_weight);

    long startTime = System.nanoTime();
    for (int i = 500; i < 1000; i++) {
      quickSel.addAssertion(assertions.get(i));
    }
    quickSel.reduceNumberOfAssertions(500);
    long elapsedTime1 = System.nanoTime() - startTime;
    quickSel.assignOptimalWeights(debug_output, constraints_weight);
    long elapsedTime2 = System.nanoTime() - startTime;

    System.out.println(String.format("Elapsed time for the last 1,000 queries: %.3f sec and %.3f sec", elapsedTime1/1e9, elapsedTime2/1e9));
  }

}
