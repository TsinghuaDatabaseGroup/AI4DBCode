package edu.illinois.quicksel.basic;

import edu.illinois.quicksel.Assertion;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Vector;

public class AssertionReader {

  static int yearSize = 123;

  static int makeSize = 24430;

  static int colorSize = 226;

  static int expireSize = 841;

  public static String assertionFileName = "test/java/edu/illinois/quicksel/resources/assertion_dmv.txt";

  public static String permanentAssertionFileName = "test/java/edu/illinois/quicksel/resources/permanent_assertion_dmv.txt";

  private static Double checkInBoundary(Double x, int boundary) {
    if (x < 0) {
      x = 0.0;
    } else if (x > boundary) {
      x = (double)boundary;
    }
    return x/boundary;
  }


  public static Pair<Vector<Assertion>, Vector<Assertion>> readAssertion() throws IOException {
    BufferedReader br = new BufferedReader(new FileReader(assertionFileName));
    String line;
    
    // left is assertion, right is permanent assertion
    Pair<Vector<Assertion>, Vector<Assertion>> assertionListPair = 
        new ImmutablePair<>(new Vector<>(), new Vector<>());
    
    while ((line = br.readLine()) != null) {
      // use comma as separator
      String[] data = line.split(",");
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      double left1, right1;
      left1 = checkInBoundary(Double.valueOf(data[0]), 1);
      right1 = checkInBoundary(Double.valueOf(data[1]), 1);
      r1.put(0, new ImmutablePair<>(left1, right1));
      double left2, right2;
      left2 = checkInBoundary(Double.valueOf(data[2]), 1);
      //left2 = ((543012*left2+19720000)-20140000)/63012;
      right2 = checkInBoundary(Double.valueOf(data[3]), 1);
      r1.put(1, new ImmutablePair<>(left2, right2));
      double left3, right3;
      left3 = checkInBoundary(Double.valueOf(data[4]), 1);
      right3 = checkInBoundary(Double.valueOf(data[5]), 1);
      //right3 = ((543012*right3+19720000)-20140000)/63012;
      r1.put(2, new ImmutablePair<>(left3, right3));
      Assertion assertion = new Assertion(r1, Double.valueOf(data[6]));
      assertionListPair.getLeft().add(assertion);
    }
    
    br.close();
    
    br = new BufferedReader(new FileReader(permanentAssertionFileName));
    while ((line = br.readLine()) != null) {
      // use comma as separator
      String[] data = line.split(",");
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      double left, right;
      left = checkInBoundary(Double.valueOf(data[0]), 1);
      right = checkInBoundary(Double.valueOf(data[1]), 1);
      r1.put(0, new ImmutablePair<>(left, right));
      left = checkInBoundary(Double.valueOf(data[2]), 1);
      right = checkInBoundary(Double.valueOf(data[3]), 1);
      r1.put(1, new ImmutablePair<>(left, right));
      double left3, right3;
      left3 = checkInBoundary(Double.valueOf(data[4]), 1);
      right3 = checkInBoundary(Double.valueOf(data[5]), 1);
      r1.put(2, new ImmutablePair<>(left3, right3));
      Assertion assertion = new Assertion(r1, Double.valueOf(data[6]));
      assertionListPair.getLeft().add(assertion);
    }
    
    br.close();
    
    return assertionListPair;
  }
  
  public static Pair<Vector<Assertion>, Vector<Assertion>> 
  readAssertion(String assertionFile) throws IOException {
    
    BufferedReader br = new BufferedReader(
        new FileReader(String.format("test/java/edu/illinois/quicksel/resources/%s", assertionFile)));
    String line;
    
    // left is assertion, right is permanent assertion
    Pair<Vector<Assertion>, Vector<Assertion>> assertionListPair 
        = new ImmutablePair<>(new Vector<>(), new Vector<>());
    
    while ((line = br.readLine()) != null) {
      // use comma as separator
      String[] data = line.split(",");
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      int columns = (data.length - 1) / 2;
      for (int i = 0; i < columns;i++) {
        double left, right;
        left = checkInBoundary(Double.valueOf(data[i*2]), 1);
        right = checkInBoundary(Double.valueOf(data[i*2+1]), 1);
        r1.put(i, new ImmutablePair<>(left, right));
      }
      Assertion assertion = new Assertion(r1, Double.valueOf(data[columns*2]));
      assertionListPair.getLeft().add(assertion);
    }
    
    br.close();
    
    return assertionListPair;
  }

  public static Pair<Vector<Assertion>, Vector<Assertion>> 
  readAssertion(String assertionFile, String permenantAssertionFile) throws IOException {
    BufferedReader br = new BufferedReader(
        new FileReader(String.format("test/java/edu/illinois/quicksel/resources/%s", assertionFile)));
    String line;
    
    // left is assertion, right is permanent assertion
    Pair<Vector<Assertion>, Vector<Assertion>> assertionListPair 
        = new ImmutablePair<>(new Vector<>(), new Vector<>());
    
    while ((line = br.readLine()) != null) {
      // use comma as separator
      String[] data = line.split(",");
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      int columns = (data.length - 1) / 2;
      for (int i = 0; i < columns;i++) {
        double left, right;
        left = checkInBoundary(Double.valueOf(data[i*2]), 1);
        right = checkInBoundary(Double.valueOf(data[i*2+1]), 1);
        r1.put(i, new ImmutablePair<>(left, right));
      }
      Assertion assertion = new Assertion(r1, Double.valueOf(data[columns*2]));
      assertionListPair.getLeft().add(assertion);
    }
    
    br.close();
    
    br = new BufferedReader(new FileReader(String.format("test/java/edu/illinois/quicksel/resources/%s", permenantAssertionFile)));
    while ((line = br.readLine()) != null) {
      // use comma as separator
      String[] data = line.split(",");
      HashMap<Integer, Pair<Double, Double>> r1 = new HashMap<>();
      int columns = (data.length - 1) / 2;
      for (int i = 0; i < columns;i++) {
        double left, right;
        left = checkInBoundary(Double.valueOf(data[i*2]), 1);
        right = checkInBoundary(Double.valueOf(data[i*2+1]), 1);
        r1.put(i, new ImmutablePair<>(left, right));
      }
      Assertion assertion = new Assertion(r1, Double.valueOf(data[columns*2]));
      assertionListPair.getRight().add(assertion);
    }
    
    br.close();
    
    return assertionListPair;
  }
}
