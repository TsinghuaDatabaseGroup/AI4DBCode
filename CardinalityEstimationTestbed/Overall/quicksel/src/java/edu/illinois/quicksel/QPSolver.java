package edu.illinois.quicksel;

import java.util.ArrayList;
import java.util.List;

public class QPSolver {

  /**
   * http://cvxopt.org/userguide/coneprog.html#quadratic-programming
   * @param P
   * @param q
   * @param G
   * @param h
   * @param A
   * @param b
   * @return
   */
  private native double[] cvxopt_solve(double[] P, double[] q, double[] G, double[] h, double[] A, double[] b);

  private native double[] cvxopt_solve(double[] P, double[] q, double[] G, double[] h);
  
  private native double[] cvxopt_solve_sparse(
      double[] PX, int[] PI, int[] PJ, double[] q, double[] GX, int[] GI, int[] GJ, double[] h);

  static {
    //        System.loadLibrary("qpsolver");
    //System.load("/Users/pyongjoo/workspace/crumbs/src/cpp/qpsolver/libqpsolver.jnilib");
    String project_home = System.getProperty("project_home");
    String platform = System.getProperty("platform");

    if (project_home != null && platform != null) {
      String platform_ext = (platform.equals("mac")) ? "jnilib" : "so";
      System.load(project_home + "/src/cpp/qpsolver/libqpsolver." + platform_ext);
    }
  }
  
  protected static double[] solve(double[][] P, double[] q, double[][] G, double[] h, double[][] A, double[] b) {		
    assert(P.length    == q.length);
    assert(P[0].length == q.length);
    assert(G.length    == q.length);
    assert(G[0].length == q.length);
    assert(A.length    == b.length);
    assert(A[0].length == q.length);

    int m = q.length;
    int n = b.length;

    double[] Pa = new double[m * m];
    double[] Ga = new double[m * m];
    double[] Aa = new double[n * m];

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        Pa[i+m*j] = P[i][j];
        Ga[i+m*j] = G[i][j];
      }
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        Aa[i+n*j] = A[i][j];
      }
    }

    QPSolver s = new QPSolver();
    return s.cvxopt_solve(Pa, q, Ga, h, Aa, b);
  }

  protected static double[] solve(double[][] P, double[] q, double[][] G, double[] h) {		
    assert(P.length == q.length);
    assert(P[0].length == q.length);
    assert(G.length == q.length);
    assert(G[0].length == q.length);

    int m = q.length;

    double[] Pa = new double[m * m];
    double[] Ga = new double[m * m];

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        Pa[i+m*j] = P[i][j];
        Ga[i+m*j] = G[i][j];
      }
    }

    QPSolver s = new QPSolver();
    return s.cvxopt_solve(Pa, q, Ga, h);
  }

  protected static double[] solve_sparse(
      double[][] P, double[] q, double[] GX, int[] GI, int[] GJ, double[] h) {
    
    assert(P.length    == q.length);
    assert(P[0].length == q.length);
    assert(GX.length   == q.length);
    assert(GI.length   == q.length);
    assert(GJ.length   == q.length);
  
    int m = q.length;
    
    // prepare P
    List<Double> PX_list = new ArrayList<Double>();
    List<Integer> PI_list = new ArrayList<Integer>();
    List<Integer> PJ_list = new ArrayList<Integer>();
    PJ_list.add(0);
    
    for (int j = 0; j < m; j++) {
      for (int i = 0; i < m; i++) {
        if (P[i][j] > 1e-4) {
          PX_list.add(P[i][j]);
          PI_list.add(i);
        }
      }
      PJ_list.add(PX_list.size());
    }
    
    int nz = PX_list.size();
    double[] PX = new double[nz];
    int[] PI = new int[nz];
    int[] PJ = new int[m+1];
    for (int i = 0; i < nz; i++) {
      PX[i] = PX_list.get(i);
      PI[i] = PI_list.get(i);
    }
    for (int i = 0; i < m+1; i++) {
      PJ[i] = PJ_list.get(i);
    }
    
    // prepare G and A
//    double[] Ga = new double[m * m];
//    for (int i = 0; i < m; i++) {
//      for (int j = 0; j < m; j++) {
//        Ga[i+m*j] = G[i][j];
//      }
//    }
  
    QPSolver s = new QPSolver();
    return s.cvxopt_solve_sparse(PX, PI, PJ, q, GX, GI, GJ, h);
  }

}
