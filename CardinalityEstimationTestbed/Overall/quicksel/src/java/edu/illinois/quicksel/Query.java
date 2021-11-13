package edu.illinois.quicksel;

import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.commons.lang3.tuple.Pair;

/**
 * May specify the ranges for a subset of dimensions.
 * 
 * @author Yongjoo Park
 */
public class Query {

  private HashMap<Integer, Pair<Double, Double>> constraints = new HashMap<Integer, Pair<Double, Double>>();

  public HashMap<Integer, Pair<Double, Double>> getConstraints() {
    return constraints;
  }

  public Query(HashMap<Integer, Pair<Double, Double>> constraints) {
    Iterator<Entry<Integer, Pair<Double, Double>>> it = constraints.entrySet().iterator();
    while (it.hasNext()) {
      Map.Entry<Integer, Pair<Double, Double>> e = it.next();
      this.constraints.put(new Integer(e.getKey()),
          Pair.of(new Double(e.getValue().getLeft()), new Double(e.getValue().getRight())));
    }
  }

  // copy constructor
  public Query(Query other) {
    this(other.getConstraints());
  }

  public boolean doesSatisfy(List<Double> data_item) {

    boolean satisfiedAll = true;

    for (int i = 0; i < data_item.size(); i++) {
      if (constraints.containsKey(Integer.valueOf(i))) {
        Pair<Double, Double> range = constraints.get(Integer.valueOf(i));
        if (data_item.get(i) < range.getLeft() || data_item.get(i) > range.getRight()) {
          satisfiedAll = false;
        }
      }
    }

    return satisfiedAll;
  }

  @Override
  public String toString() {
    return constraints.toString();
  }

  /**
   * 
   * @param min_max The domain of a dataset.
   * @return
   */
  public Hyperrectangle rectangleFromQuery(Hyperrectangle min_max) {
    Hyperrectangle rec = new Hyperrectangle(min_max);

    for (Map.Entry<Integer, Pair<Double, Double>> e : this.getConstraints().entrySet()) {
      Integer k = e.getKey();
      Pair<Double, Double> v = e.getValue();
      rec.intervals.set(k, Pair.of(v.getLeft(), v.getRight()));
    }

    return rec;
  }
}
