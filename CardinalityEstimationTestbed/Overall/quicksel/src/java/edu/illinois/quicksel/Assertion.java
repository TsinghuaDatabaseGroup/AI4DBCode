package edu.illinois.quicksel;

import java.util.HashMap;
import org.apache.commons.lang3.tuple.Pair;

/**
 * A pair of (query, selectivity)
 * @author Yongjoo Park
 *
 */
public class Assertion {

  public Query query;

  public double freq;

  public Assertion(Query query, double freq) {
    this.query = query;
    this.freq = freq;
  }

  public Assertion(HashMap<Integer, Pair<Double, Double>> constraints, double freq) {
    this.query = new Query(constraints);
    this.freq = freq;
  }

  @Override
  public String toString() {
    String ret = "";
    ret += query.toString() + ": " + String.valueOf(freq);
    return ret;
  }
}
