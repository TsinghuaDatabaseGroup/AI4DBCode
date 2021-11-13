package edu.illinois.quicksel.experiments;

import edu.illinois.quicksel.basic.SyntheticDataGenerator;
import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.quicksel.QuickSel;
import edu.illinois.quicksel.peratt.PerAtt;
import edu.illinois.quicksel.sampling.Sampling;
import org.apache.commons.lang3.tuple.Pair;

import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PerAttAndSamplingTest {

  public static List<Integer> list = Arrays.asList(10,
      20,
      30,
      40,
      50,
      60,
      70,
      80,
      90,
      100,
      200,
      300,
      400,
      500,
      600,
      700,
      800,
      900,
      1000);

  public static void main(String[] args) throws FileNotFoundException {

    int assertion_count = 1000;
    int test_query_num = 1000;          // total number is test_query_num

    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(0.5, 10000);
    List<Assertion> permanent_assertions = SyntheticDataGenerator.twoDimDefaultAssertion(dataset);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(assertion_count, dataset);
    List<Assertion> queryset = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(test_query_num, dataset);

    System.out.println("dataset and query set generations done.\n");

    System.out.println("PerAtt test");
    perAttTest(dataset, queryset);
    System.out.println("");

    System.out.println("Sampling test");
    samplingTest(dataset, queryset);
    System.out.println("");

    System.out.println("Crumbs test");
    quickSelTest(dataset, permanent_assertions, assertions, queryset);
    System.out.println("");

  }

  private static void quickSelTest(
      List<List<Double>> dataset,
      List<Assertion> permanent_assertions,
      List<Assertion> assertions,
      List<Assertion> queryset) {
    for (int assertionNum : list) {
      double error = 0;

      Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(dataset);
      QuickSel quickSel = new QuickSel(range_freq.getLeft(), range_freq.getRight());

      for (Assertion a : permanent_assertions) {
        quickSel.addPermanentAssertion(a);
      }

      for (Assertion a : assertions.subList(0, assertionNum)) {
        quickSel.addAssertion(a);
      }
      quickSel.prepareOptimization();

      boolean debug_output = true;
      quickSel.assignOptimalWeights(debug_output);

      double error_sum = 0.0;
      for (Assertion a : queryset) {
        Query q = a.query;
        double est = quickSel.answer(q);
        double actual = a.freq;
        error_sum += Math.pow(est - actual, 2);
      }
      error += Math.sqrt(error_sum / (double) queryset.size());
      System.out.println(String.format("Learning %d assertions, RMS error: %.5f\n", assertionNum, error));
    }
  }

  private static void perAttTest(
      List<List<Double>> dataset,
      List<Assertion> queryset) {
    List<Pair<Double, Double>> min_max = new ArrayList<>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    for (int assertionNum : list) {
      double error = 0;

      PerAtt perAtt = new PerAtt(dataset, (int) Math.floor(Math.sqrt(assertionNum)), min_max_rec, 1.0);

      double error_sum = 0.0;
      for (Assertion a : queryset) {
        Query q = a.query;
        double est = perAtt.answer(q);
        double actual = a.freq;
        error_sum += Math.pow(est - actual, 2);
      }
      error += Math.sqrt(error_sum / (double) queryset.size());

      System.out.println(String.format("Learning %d assertions, RMS error: %.5f", assertionNum, error));
    }
  }

  private static void samplingTest(
      List<List<Double>> dataset,
      List<Assertion> queryset) {
    List<Pair<Double, Double>> min_max = new ArrayList<>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    for (int assertionNum : list) {
      double error = 0;
      Sampling sampling = new Sampling(dataset, assertionNum, min_max_rec, 1.0);

      double error_sum = 0.0;
      for (Assertion a : queryset) {
        Query q = a.query;
        double est = sampling.answer(q);
        double actual = a.freq;
        error_sum += Math.pow(est - actual, 2);
      }
      error += Math.sqrt(error_sum / (double) queryset.size());

      System.out.println(String.format("Learning %d assertions, RMS error: %.5f", assertionNum, error));
    }
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
