package edu.illinois.quicksel.quicksel;

import java.sql.Timestamp;
import java.util.Arrays;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;
import org.jblas.ranges.IntervalRange;

/**
 * 
 * Reference: https://www.cs.ccu.edu.tw/~wtchu/courses/2015s_OPT/Lectures/Chapter%2023%20Algorithms%20for%20Constrained%20Optimization.pdf
 * 
 * @param l constraint weight
 */
public class ProjectionAlgorithm {
  
  public static double[] solve(DoubleMatrix P1, DoubleMatrix A1, DoubleMatrix b1, double l) {
    int max_iter = 20;
    return solve(P1, A1, b1, l, max_iter);
  }
  
  public static double[] solve(DoubleMatrix P1, DoubleMatrix A1, DoubleMatrix b1, double l, int max_iter) {
    
    System.out.println(new Timestamp(System.currentTimeMillis()) +  " Optimization routine starts");
    
    int n = b1.length;
    int m = P1.columns;
    System.out.println("# of internal variables: " + m);
    System.out.println("# of assertions: " + n);
    
    double weight_on_sum_to_one = 10.0;
    DoubleMatrix Asub = A1.getRows(new IntervalRange(1, n));
    DoubleMatrix bsub = b1.getRows(new IntervalRange(1, n));
    DoubleMatrix one = DoubleMatrix.ones(m,1);
    double total_freq = b1.get(0);
    
    // factor to obtain x
    DoubleMatrix x0 = Solve.solve(
        P1.add(Asub.transpose().mmul(Asub).muli(l / weight_on_sum_to_one))
          .add(one.mmul(one.transpose()).muli(l)),
          Asub.transpose().mmul(bsub).muli(l / weight_on_sum_to_one)
          .add(one.mul(l).mul(total_freq)));
    DoubleMatrix xt = new DoubleMatrix(Arrays.copyOf(x0.data, x0.length));
    
    // this part does not seem to work, so we disable
    for (int t = 0; t < max_iter; t++) {
      DoubleMatrix gap = A1.mmul(xt).sub(b1);
      System.out.println(String.format("[%d] avg l1 gap: %.3f", t, gap.norm1() / xt.length));
      
      double step_size = 1.0 / (double) (t+2);
      
      // projection to a feasible region
      xt.maxi(0.0);
      
      // gradient step
      DoubleMatrix d = xt.sub(x0);
      xt.subi(d.mul(step_size));
    }
    
    System.out.println(new Timestamp(System.currentTimeMillis()) +  " Optimization done.");
    
    return xt.data;
  }

}
