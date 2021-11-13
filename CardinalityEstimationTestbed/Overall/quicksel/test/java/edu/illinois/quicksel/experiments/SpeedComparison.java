package edu.illinois.quicksel.experiments;

import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;

import edu.illinois.quicksel.basic.SyntheticDataGenerator;
import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.quicksel.QuickSel;
import edu.illinois.quicksel.isomer.Isomer;

public class SpeedComparison {

  public static void main(String[] args) {

    int assertion_count = 500;
    int test_query_num = 1000;					// total number is test_query_num

    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> permanent_assertions = SyntheticDataGenerator.twoDimDefaultAssertion(dataset);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions2(assertion_count, dataset);
    List<Assertion> queryset = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions2(test_query_num, dataset);

    System.out.println("dataset and query set generations done.\n");

    System.out.println("QuickSel test");
    quickSelTest(dataset, permanent_assertions, assertions, queryset);
    System.out.println("");

    System.out.println("Isomer test");
    isomerTest(dataset, permanent_assertions, assertions, queryset);
    System.out.println("");

    System.out.println("Isomer+QP test");
    isomerQPTest(dataset, permanent_assertions, assertions, queryset);
    System.out.println("");
  }

  private static void quickSelTest(
      List<List<Double>> dataset,
      List<Assertion> permanent_assertions,
      List<Assertion> assertions,
      List<Assertion> queryset) {

    // build Crumbs
    Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(dataset);
    QuickSel quickSel = new QuickSel(range_freq.getLeft(), range_freq.getRight());
//    crumbs.setEnforcedVarCount(20);
    
//    System.out.println("Assertions:");
//    for (Assertion a : assertions) {
//      System.out.println(a.query.rectangleFromQuery(range_freq.getLeft()));
//    }
//    System.out.println();

    for (Assertion a : permanent_assertions) {
      quickSel.addPermanentAssertion(a);
    }

    long time1 = System.nanoTime();
    for (Assertion a : assertions) {
      quickSel.addAssertion(a);
    }
    quickSel.prepareOptimization();
    long time2 = System.nanoTime();

    System.out.println("Optimization starts: " + new Timestamp(System.currentTimeMillis()));
    boolean debug_output = true;
    quickSel.assignOptimalWeights(debug_output);
    long time3 = System.nanoTime();
    System.out.println("Optimization done: " + new Timestamp(System.currentTimeMillis()));

    double error_sum = 0.0;
    for (Assertion a : queryset) {
      Query q = a.query;
      double est = quickSel.answer(q);
      double actual = a.freq;
      error_sum += Math.pow(est - actual, 2);
    }
    long time4 = System.nanoTime();
    
//    System.out.println("Kernels:");
//    crumbs.printKernels();
    
    System.out.println(String.format("Test rms error: %.4f", Math.sqrt(error_sum / (double) queryset.size())));

    System.out.println(String.format("Insertion time for %d queries: %.3f sec", assertions.size(), (time2 - time1) / 1e9));
    System.out.println(String.format("Optimization time for %d queries: %.3f sec", assertions.size(), (time3 - time2) / 1e9));
    System.out.println(String.format("Estimation time for %d queries: %.3f sec", queryset.size(), (time4 - time3) / 1e9));
  }

  private static void isomerTest(
      List<List<Double>> dataset,
      List<Assertion> permanent_assertions,
      List<Assertion> assertions,
      List<Assertion> queryset) {

    // build Crumbs
    Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(dataset);
    Isomer isomer = new Isomer(range_freq.getLeft(), range_freq.getRight());

    long time1 = System.nanoTime();
    for (Assertion a : assertions) {
      isomer.addAssertion(a);
    }
    long time2 = System.nanoTime();

    boolean debug_output = true;
    long time3 = System.nanoTime();

    double error_sum = 0.0;
    for (Assertion a : queryset) {
      Query q = a.query;
      double est = isomer.answer(q);
      double actual = a.freq;
      error_sum += Math.pow(est - actual, 2);
    }
    long time4 = System.nanoTime();

    System.out.println(String.format("Test rms error: %.4f", Math.sqrt(error_sum / (double) queryset.size())));
    System.out.println(String.format("Insertion time for %d queries: %.3f sec", assertions.size(), (time2 - time1) / 1e9));
    System.out.println(String.format("Optimization time for %d queries: %.3f sec", assertions.size(), (time3 - time2) / 1e9));
    System.out.println(String.format("Estimation time for %d queries: %.3f sec", queryset.size(), (time4 - time3) / 1e9));
  }
  
  private static void isomerQPTest(
      List<List<Double>> dataset,
      List<Assertion> permanent_assertions,
      List<Assertion> assertions,
      List<Assertion> queryset) {

    // build Crumbs
    Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(dataset);
    Isomer isomer = new Isomer(range_freq.getLeft(), range_freq.getRight());

    long time1 = System.nanoTime();
    for (Assertion a : assertions) {
      isomer.addAssertion(a);
    }
    long time2 = System.nanoTime();

    boolean debug_output = true;
    //isomer.assignOptimalWeightsQP(debug_output);
    long time3 = System.nanoTime();

//    for (Query q : queryset) {
//      isomer.answer(q);
//    }
    double error_sum = 0.0;
    for (Assertion a : queryset) {
      Query q = a.query;
      double est = isomer.answer(q);
      double actual = a.freq;
      error_sum += Math.pow(est - actual, 2);
    }
    long time4 = System.nanoTime();

    System.out.println(String.format("Test rms error: %.4f", Math.sqrt(error_sum / (double) queryset.size())));
    System.out.println(String.format("Insertion time for %d queries: %.3f sec", assertions.size(), (time2 - time1) / 1e9));
    System.out.println(String.format("Optimization time for %d queries: %.3f sec", assertions.size(), (time3 - time2) / 1e9));
    System.out.println(String.format("Estimation time for %d queries: %.3f sec", queryset.size(), (time4 - time3) / 1e9));
  }

  private static Pair<Hyperrectangle, Double> computeMinMaxRange(List<List<Double>> dataset) {
    List<Pair<Double, Double>> min_max = new ArrayList<>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = SyntheticDataGenerator.countSatisfyingItems(dataset, min_max_rec.toQuery()) / ((double) dataset.size());
    return Pair.of(min_max_rec, total_freq);
  }

}
