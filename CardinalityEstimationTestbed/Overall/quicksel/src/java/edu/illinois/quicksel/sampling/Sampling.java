package edu.illinois.quicksel.sampling;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.SelectivityEstimator;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 
 * @author Boyu
 *
 */
public class Sampling implements SelectivityEstimator {

  private List<List<Double>> samples = new ArrayList<>();
  
  private Hyperrectangle min_max;
  
  private double totFreq;

  public Sampling(List<List<Double>> dataset, int numSamples, Hyperrectangle min_max, double totFreq) {
    this.min_max = new Hyperrectangle(min_max);
    this.totFreq = totFreq;
    Random rand = new Random();
    for (int i = 0; i < numSamples; ++i) {
      int r = rand.nextInt(dataset.size());
      samples.add(new ArrayList<Double>(dataset.get(r)));
    }
  }

  public Sampling(List<List<Double>> dataset, int numSamples, Hyperrectangle min_max) {
    this(dataset, numSamples, min_max, 1.0);
  }

  private boolean isInRange(List<Double> pt, Hyperrectangle rec) {
    for (int i = 0; i < pt.size(); ++i) {
      if (pt.get(i) < rec.intervals.get(i).getLeft()
          || pt.get(i) > rec.intervals.get(i).getRight()) {
        return false;
      }
    }

    return true;
  }

  public double answer(Query q) {
    Hyperrectangle rec = q.rectangleFromQuery(min_max);
    int cnt = 0;
    for (List<Double> pt : samples) {
      if (isInRange(pt, rec)) {
        ++cnt;
      }
    }
    return cnt / (double) samples.size() * totFreq;
  }
}
