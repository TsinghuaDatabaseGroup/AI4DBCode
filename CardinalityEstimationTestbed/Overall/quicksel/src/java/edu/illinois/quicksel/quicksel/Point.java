package edu.illinois.quicksel.quicksel;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.ml.clustering.Clusterable;

public class Point implements Clusterable {
  
  private double[] p;
  
  public Point(List<Double> point) {
    double[] p = new double[point.size()];
    for (int i = 0; i < point.size(); i++) {
      p[i] = point.get(i);
    }
    this.p = p;
  }

  @Override
  public double[] getPoint() {
    return p;
  }
  
  public double distanceTo(Clusterable other) {
    double[] op = other.getPoint();
    double dsum = 0.0;
    for (int i = 0; i < p.length; i++) {
      dsum += Math.pow(p[i] - op[i], 2);
    }
    return Math.sqrt(dsum / (double) p.length);
  }
  
  public List<Double> perDimDistance(Clusterable other) {
    double[] op = other.getPoint();
    List<Double> perDimDist = new ArrayList<>();
    for (int i = 0; i < p.length; i++) {
      perDimDist.add(Math.abs(p[i] - op[i]));
    }
    return perDimDist;
  }

}
