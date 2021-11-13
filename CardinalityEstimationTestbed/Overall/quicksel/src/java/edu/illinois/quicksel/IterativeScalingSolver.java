package edu.illinois.quicksel;

import java.util.Vector;

/**
 * The solver used for ISOMER
 * @author Yongjoo Park
 *
 */
public class IterativeScalingSolver {

  public static double[] solve(double[][] A, double[] b, double[] v) {
    return solve(A, b, v, 1e-5, 10000);
  }

  /**
   * 
   * @param A Matrix of shape (n, m)
   * @param b Column vector of length n
   * @param v Column vector of length m
   * @param e This constant determines if this algorithm converged
   * @param maxiters The maximum number of iterations to take
   * @return Optimal weights (vector of length m)
   */
  public static double[] solve(double[][] A, double[] b, double[] v, double e, int maxiters) {
    int n = A.length;	assert(n > 0);
    int m = A[0].length;
    double[] z = new double[n];		// exp(a_j * \lambda_j), where \lambda_j is the Lagrange multiplier for x[j]
    for (int i = 0; i < n; i++) {
      z[i] = 1.0;
    }

    // index for non-zero entries
    Vector<Vector<Integer>> I = new Vector<Vector<Integer>>();
    for (int i = 0; i < n; i++) {
      Vector<Integer> Ii = new Vector<Integer>();
      for (int j = 0; j < m; j++) {
        if (A[i][j] > 0) {
          Ii.add(j);
        }
      }
      I.add(Ii);
    }

    Vector<Vector<Integer>> J = new Vector<Vector<Integer>>();
    for (int j = 0; j < m; j++) {
      Vector<Integer> Ji = new Vector<Integer>();
      for (int i = 0; i < n; i++) {
        if (A[i][j] > 0) {
          Ji.add(i);
        }
      }
      J.add(Ji);
    }


    // main module
    double new_delta = e;
    double old_delta = 10 * e;


    // update rule
    // z[i] = b[i] e / (\sum_{j=1}^m v[j] a[i][j] \prod_{k=1}^n z[k]^a[k][j] / z[i])
    for (int it = 0; it < maxiters; it++) {
      if (Math.abs(new_delta - old_delta) <= e) {
        break;
      }
      old_delta = new_delta;
      new_delta = 0.0;

      for (int i = 0; i < n; i++) {	// update z[i]
        double denom = 0;

        for (Integer j : I.get(i)) {
          double zj = 1.0;

          for (Integer k : J.get(j)) {
            zj *= Math.pow(z[k], A[k][j]);
          }
          denom += v[j] * A[i][j] * zj / z[i];
        }

        double zn = b[i] * Math.exp(1) / denom;
        new_delta += Math.abs(z[i] / zn);
        z[i] = zn;
      }

      if (it % 10 == 0 && it != 0) {
        System.out.println(String.format("iter count: %d, delta gap: %f", it, Math.abs(new_delta - old_delta)));
      }
    }

    // set output
    double[] w = new double[m];
    for (int i = 0; i < m; i++) {
      w[i] = 1.0;
    }
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < n; i++) {
        w[j] *= Math.pow(z[i], A[i][j]);
      }
      w[j] *= v[j] / Math.exp(1);
    }

    return w;
  }
}
