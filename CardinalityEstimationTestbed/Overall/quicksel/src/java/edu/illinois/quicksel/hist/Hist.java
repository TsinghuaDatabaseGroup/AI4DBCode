package edu.illinois.quicksel.hist;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.SelectivityEstimator;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static java.lang.Math.max;
import static java.lang.Math.min;


public class Hist implements SelectivityEstimator {

    private int numBinPerAtt;

    private Vector<Double> binFreq = new Vector<Double>();

    private Vector<Hyperrectangle> bins = new Vector<>();

    private Hyperrectangle min_max;

    private double totFreq;

    private int dimension;

    public Hist(List<List<Double>> data, int numBinPerAtt, Hyperrectangle min_max, double totFreq) {
        this.dimension = data.get(0).size();
        this.numBinPerAtt = numBinPerAtt;
        this.min_max = new Hyperrectangle(min_max);
        this.totFreq = totFreq;

        // create bins
        List<List<Pair<Double, Double>>> bins = addBins(min_max);
        for (List<Pair<Double, Double>> interval : bins) {
            Hyperrectangle bin = new Hyperrectangle(interval);
            this.bins.add(bin);
            this.binFreq.add(getFreq(data, bin));
        }

        //System.out.println(this.bins.size());
    }

    public Hist(List<List<Double>> data, int numBinPerAtt, Hyperrectangle min_max) {
        this(data, numBinPerAtt, min_max, 1.0);
    }


    private List<List<Pair<Double, Double>>> addBins(Hyperrectangle min_max) {
        List<List<Pair<Double, Double>>> bins = new ArrayList<>();
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < numBinPerAtt; j++) {
                List<Pair<Double, Double>> intervals = new ArrayList<>();
                for (int k = 0; k < dimension; k++) {
                    if (k == i) {
                        double left = min_max.intervals.get(i).getLeft();
                        double right = min_max.intervals.get(i).getRight();
                        double binSize = (right - left) / numBinPerAtt;
                        intervals.add(new ImmutablePair<>(left + j * binSize, left + (j + 1) * binSize));
                    } else {
                        intervals.add(new ImmutablePair<>(0.0, 1.0));
                    }
                }
                bins.add(intervals);
            }

        }
        return bins;
    }

    private double getFreq(List<List<Double>> data, Hyperrectangle bin) {
        int cnt = data.size();
        for (List<Double> point : data) {
            // check point inside the bin
            for (int idx = 0; idx < bin.intervals.size(); idx++) {
                Pair<Double, Double> interval = bin.intervals.get(idx);
                // make sure pts inside boundary
                if (point.get(idx) < 0) {
                    point.set(idx, 0.0);
                } else if (point.get(idx) > 1) {
                    point.set(idx, 1.0);
                }
                if (point.get(idx) < interval.getLeft() || point.get(idx) > interval.getRight()) {
                    cnt--;
                    break;
                }
            }
        }
        return (double) cnt / (double) data.size();
    }

    public double answer(Query q) {
        Hyperrectangle query = q.rectangleFromQuery(min_max);

        double result = 1.0;

        for (int column = 0; column < bins.size() / numBinPerAtt; column++) {
            double colFreq = 0.0;
            for (int i = 0; i < numBinPerAtt; i++) {
                Hyperrectangle bin = bins.get(column * numBinPerAtt + i);
                double intersection = intersect(bin, query, column);
                colFreq += intersection / bin.vol() * binFreq.get(column * numBinPerAtt + i);
            }
            result *= colFreq;
        }

        return result;
    }

    private double intersect(Hyperrectangle bin, Hyperrectangle other, int column) {
        Pair<Double, Double> p = bin.intervals.get(column);
        double a = p.getLeft();
        double b = p.getRight();

        Pair<Double, Double> o = other.intervals.get(column);
        double c = o.getLeft();
        double d = o.getRight();

        return computeInterval(a, b, c, d);
    }

    private double computeInterval(double a, double b, double c, double d) {
        assert (a < b);
        assert (c < d);
        if (b <= c) return 0;
        if (d <= a) return 0;
        return min(b, d) - max(a, c);
    }

}
