package edu.illinois.quicksel.peratt;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.SelectivityEstimator;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

/**
 * Histogram per each dimension.
 *
 * @author Boyu
 */
public class PerAtt implements SelectivityEstimator {

  private int numBinPerAtt;

  private Vector<Double> binFreq = new Vector<Double>();

  private Vector<Hyperrectangle> bins = new Vector<>();

  private Hyperrectangle min_max;

  private double totFreq;

  public PerAtt(List<List<Double>> data, int numBinPerAtt, Hyperrectangle min_max, double totFreq) {

    this.numBinPerAtt = numBinPerAtt;
    this.min_max = new Hyperrectangle(min_max);
    this.totFreq = totFreq;

    // create bins
    List<List<Pair<Double, Double>>> bins = addBins(min_max, 0, new ArrayList<>());
    for (List<Pair<Double, Double>> interval:bins) {
      Hyperrectangle bin = new Hyperrectangle(interval);
      this.bins.add(bin);
      this.binFreq.add(getFreq(data, bin));
    }
  }

  public PerAtt(List<List<Double>> data, int numBinPerAtt, Hyperrectangle min_max) {
    this(data, numBinPerAtt, min_max, 1.0);
  }


  // recursively generate intervals of bins
  private List<List<Pair<Double, Double>>> addBins(Hyperrectangle min_max, int dimension, List<List<Pair<Double, Double>>> bins) {
    List<List<Pair<Double, Double>>> new_bins = new ArrayList<>();

    if (dimension == min_max.d) {
      return bins;
    } else if (dimension == 0) {
      double left = min_max.intervals.get(dimension).getLeft();
      double right = min_max.intervals.get(dimension).getRight();
      double binSize = (right - left) / numBinPerAtt;
      for (int j = 0; j < numBinPerAtt; j++) {
        List<Pair<Double, Double>> intervals = new ArrayList<>();
        intervals.add(new ImmutablePair<>(left + j * binSize, left + (j + 1) * binSize));
        new_bins.add(intervals);
      }
    } else {
      double left = min_max.intervals.get(dimension).getLeft();
      double right = min_max.intervals.get(dimension).getRight();
      double binSize = (right - left) / numBinPerAtt;
      for (int i = 0; i < bins.size(); i++) {
        for (int j = 0; j < numBinPerAtt; j++) {
          List<Pair<Double, Double>> intervals = new ArrayList<>(bins.get(i));
          intervals.add(new ImmutablePair<>(left + j * binSize, left + (j + 1) * binSize));
          new_bins.add(intervals);
        }
      }
    }
    return addBins(min_max, dimension + 1, new_bins);
  }

  private double getFreq(List<List<Double>> data, Hyperrectangle bin) {
    int cnt = data.size();
    for (List<Double> point : data) {
      // check point inside the bin
      for (int idx=0; idx<bin.intervals.size(); idx++) {
        Pair<Double, Double> interval = bin.intervals.get(idx);
        // make sure pts inside boundary
        if (point.get(idx)<0) {
          point.set(idx, 0.0);
        } else if (point.get(idx)>1) {
          point.set(idx, 1.0);
        }
        if (point.get(idx)<interval.getLeft() || point.get(idx)>interval.getRight()) {
          cnt--;
          break;
        }
      }
    }
    return (double) cnt / (double) data.size();
  }

  public double answer(Query q) {
    Hyperrectangle query = q.rectangleFromQuery(min_max);

    double result = 0.0;

    for (int i=0;i<bins.size(); i++) {
      Hyperrectangle bin = bins.get(i);
      result += bin.intersect(query) / bin.vol() * binFreq.get(i);
    }

    return result;
  }

}
