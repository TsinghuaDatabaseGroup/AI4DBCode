package edu.illinois.quicksel.experiments;

import edu.illinois.quicksel.basic.AssertionReader;
import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.quicksel.QuickSel;
import edu.illinois.quicksel.isomer.Isomer;
import org.apache.commons.lang3.tuple.Pair;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Vector;
import java.io.File;

public class JOBSpeedComparison {

  public static void main(String[] args) throws IOException {

    File file = new File("/Users/jintao/Work/quicksel/test/java/edu/illinois/quicksel/resources/JOB/cols-sql/2/test/");
    String[] children = file.list();
    for (String filename: children) {
      if (filename.endsWith("assertion")) {
        System.out.println(filename);
        Pair<Vector<Assertion>, Vector<Assertion>> assertionPair = AssertionReader.readAssertion("JOB/cols-sql/2/train/"+filename, "JOB/cols-sql/2/train/"+filename.split("\\.")[0]+".permanent");
        Vector<Assertion> assertions = assertionPair.getLeft();
        Vector<Assertion> permanent_assertions = assertionPair.getRight();
        Pair<Vector<Assertion>, Vector<Assertion>> test_assertionPair = AssertionReader.readAssertion("JOB/cols-sql/2/test/"+filename, "JOB/cols-sql/2/test/"+filename.split("\\.")[0]+".permanent");
        Vector<Assertion> queryAssertion = test_assertionPair.getLeft();

        System.out.println("dataset and query set generations done.\n");

        System.out.println("QuickSel test");
        quickSelTest(permanent_assertions, assertions, queryAssertion);
        System.out.println("");
      }
    }

    // System.out.println("Isomer test");
    // isomerTest(permanent_assertions, assertions, queryAssertion);
    // System.out.println("");
  }

  private static void quickSelTest(
      Vector<Assertion> permanent_assertions,
      Vector<Assertion> assertions,
      List<Assertion> queryset) {

    // build Crumbs
//    List<Integer> list = Arrays.asList(10,
//        20,
//        30,
//        50,
//        100,
//        200,
//        300,
//        500);
    System.out.println(assertions.size());
//    List<Integer> list = Arrays.asList(assertions.size());
    List<Integer> list = Arrays.asList(10000);
    for (int assertionNum : list) {
      Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(assertions.firstElement());
      QuickSel quickSel = new QuickSel(range_freq.getLeft(), range_freq.getRight());

      for (Assertion a : permanent_assertions) {
        quickSel.addPermanentAssertion(a);
      }

      long time1 = System.nanoTime();
      for (Assertion a : assertions.subList(0, assertionNum)) {
        quickSel.addAssertion(a);
      }
      quickSel.prepareOptimization();
      long time2 = System.nanoTime();

      boolean debug_output = false;
      quickSel.assignOptimalWeights(debug_output);
      long time3 = System.nanoTime();

      for (Assertion q : queryset) {
        quickSel.answer(q.query);
      }
      long time4 = System.nanoTime();

      //write time
      System.out.println(String.format("Insertion time: %.3f, Optimization time: %.3f, Estimation time: %.3f", (time2 - time1) / 1e9, (time3 - time2) / 1e9, (time4 - time3) / 1e9));

      //write sel
      double squared_err_sum = 0.0;
      for (Assertion q : queryset) {
        Double sel = Math.max(0, quickSel.answer(q.query));
        squared_err_sum += Math.pow(sel - q.freq, 2);
      }
      double rms_err = Math.sqrt(squared_err_sum / queryset.size());

      System.out.println(String.format("Learning %d assertions, RMS error: %.5f\n", assertionNum, rms_err));
    }
  }

  private static void isomerTest(
      Vector<Assertion> permanent_assertions,
      Vector<Assertion> assertions,
      List<Assertion> queryset) {
    List<Integer> list = Arrays.asList(10,
        20,
        30,
        50,
        100);
    for (int assertionNum : list) {
      Pair<Hyperrectangle, Double> range_freq = computeMinMaxRange(assertions.firstElement());
      Isomer isomer = new Isomer(range_freq.getLeft(), range_freq.getRight());

      long time1 = System.nanoTime();
      for (Assertion a : assertions.subList(0, assertionNum)) {
        isomer.addAssertion(a);
      }
      long time2 = System.nanoTime();

      boolean debug_output = false;
      isomer.assignOptimalWeights(debug_output);
      long time3 = System.nanoTime();

      for (Assertion q : queryset) {
        isomer.answer(q.query);
      }
      long time4 = System.nanoTime();

      //write time
      System.out.println(String.format("Insertion time: %.3f, Optimization time: %.3f, Estimation time: %.3f", (time2 - time1) / 1e9, (time3 - time2) / 1e9, (time4 - time3) / 1e9));

      //write sel
      double squared_err_sum = 0.0;
      for (Assertion q : queryset) {
        Double sel = Math.max(0, isomer.answer(q.query));
        squared_err_sum += Math.pow(sel - q.freq, 2);
      }
      double rms_err = Math.sqrt(squared_err_sum / queryset.size());
      System.out.println(String.format("Learning %d assertions, RMS error: %.5f\n", assertionNum, rms_err));
    }
  }


  private static Pair<Hyperrectangle, Double> computeMinMaxRange(Assertion q) {
    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    for (int i = 0; i < q.query.getConstraints().size(); i ++) {
      min_max.add(Pair.of(0.0, 1.0));
    }
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = 1.0;
    return Pair.of(min_max_rec, total_freq);
  }

}
