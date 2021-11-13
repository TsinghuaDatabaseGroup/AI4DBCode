package edu.illinois.quicksel;

import org.jblas.DoubleMatrix;

public class OptimizerWrapper {

  public static double constraint_penalty = 1e10;

  public static double[] safe_squared_norm_solver(double[][] A1, double[] b1, double[] v1) {
    return safe_squared_norm_solver(A1, b1, v1, constraint_penalty);
  }

  /**
   * (no) Preferred optimization routine for Crumbs.
   * @param A1 Encodes the overlap between past queries and uniform distributions in a mixture model; A matrix of size (n, m)
   * @param b1 Answers to past queries; A column vector of length 'n'
   * @param v1 Volumes of the uniform distributions; A column vector of length 'm'
   * @param l The weights on the constraints
   * @return The weights of the uniform distributions in a mixture model
   */
  public static double[] safe_squared_norm_solver(double[][] A1, double[] b1, double[] v1, double l) {

    int n = A1.length;
    int m = A1[0].length;

    DoubleMatrix P = DoubleMatrix.zeros(m, m);
    DoubleMatrix q = DoubleMatrix.zeros(m);
    DoubleMatrix A = new DoubleMatrix(A1);
    DoubleMatrix b = new DoubleMatrix(b1);
    DoubleMatrix v = new DoubleMatrix(v1);

    for (int i = 0; i < m; i++) {
      P.put(i, i, 1 / v.get(i));
    }
    P.addi(A.transpose().mmul(A).muli(l));
    q.addi(A.transpose().mmul(b)).muli(-l);

    // positivity constraints
    double[][] G = new double[m][m];
    double[] h = new double[m];
    for (int i = 0; i < m; i++) {
      G[i][i] = -1.0;
    }

    return QPSolver.solve(Helpers.jblasMatrixToArray(P), Helpers.jblasVectorToArray(q), G, h);
  }

  public static double[] safe_squared_norm_solver(DoubleMatrix P1, DoubleMatrix A1, DoubleMatrix b1) {
    return safe_squared_norm_solver(P1, A1, b1, constraint_penalty);
  }
  
  public static double[] safe_squared_norm_solver_sparse(DoubleMatrix P1, DoubleMatrix A1, DoubleMatrix b1) {
    return safe_squared_norm_solver_sparse(P1, A1, b1, constraint_penalty);
  }

  /**
   * 
   * @param P (m,m)-matrix; pairwise overlaps between kernels.
   * @param A (n,m): intersection between queries and kernels.
   * @param b freq (n,1)
   * @param l penalty
   * @return
   */
  public static double[] safe_squared_norm_solver(DoubleMatrix P, DoubleMatrix A, DoubleMatrix b, double l) {
    int n = A.rows;
    int m = A.columns;

    DoubleMatrix q = DoubleMatrix.zeros(m);

    P.addi(A.transpose().mmul(A).muli(l));
    q.addi(A.transpose().mmul(b).muli(-l));

    // positivity
    double[][] G = new double[m][m];
    double[] h = new double[m];
    for (int i = 0; i < m; i++) {
      G[i][i] = -1.0;
    }

    return QPSolver.solve(Helpers.jblasMatrixToArray(P), Helpers.jblasVectorToArray(q), G, h);
  }
  
  /**
   * This currently does not work due to some unknown problems in C part
   * @param P
   * @param A
   * @param b
   * @param l
   * @return
   */
  public static double[] safe_squared_norm_solver_sparse(DoubleMatrix P, DoubleMatrix A, DoubleMatrix b, double l) {
    int n = A.rows;
    int m = A.columns;

    DoubleMatrix q = DoubleMatrix.zeros(m);

    P.addi(A.transpose().mmul(A).muli(l));
    q.addi(A.transpose().mmul(b)).muli(-l);

    // positivity
    double[] GX = new double[m];
    int[] GI = new int[m];
    int[] GJ = new int[m+1];
    GJ[0] = 0;
    double[] h = new double[m];
    for (int i = 0; i < m; i++) {
      GX[i] = -1.0;
      GI[i] = i;
      GJ[i+1] = i;
    }

    return QPSolver.solve_sparse(Helpers.jblasMatrixToArray(P), Helpers.jblasVectorToArray(q), GX, GI, GJ, h);
  }

  public static double[] squared_norm_solver(DoubleMatrix P, DoubleMatrix A, DoubleMatrix b) {
    int n = b.rows;
    int m = P.columns;

    DoubleMatrix q = DoubleMatrix.zeros(m);

    double[][] G = new double[m][m];
    double[] h = new double[m];
    for (int i = 0; i < m; i++) {
      G[i][i] = -1.0;
    }

    return QPSolver.solve(Helpers.jblasMatrixToArray(P), Helpers.jblasVectorToArray(q), G, h, Helpers.jblasMatrixToArray(A), Helpers.jblasVectorToArray(b));
  }


  public static double[] squared_norm_solver(double[][] A1, double[] b1, double[] v1) {
    int n = A1.length;
    int m = A1[0].length;

    DoubleMatrix P = DoubleMatrix.zeros(m, m);
    DoubleMatrix q = DoubleMatrix.zeros(m);
    DoubleMatrix v = new DoubleMatrix(v1);

    for (int i = 0; i < m; i++) {
      P.put(i, i, 1 / v.get(i));
    }

    double[][] G = new double[m][m];
    double[] h = new double[m];
    for (int i = 0; i < m; i++) {
      G[i][i] = -1.0;
    }

    return QPSolver.solve(Helpers.jblasMatrixToArray(P), Helpers.jblasVectorToArray(q), G, h, A1, b1);
  }

  public static double[] entropy_solver(double[][] A, double[] b, double[] v) {
    return IterativeScalingSolver.solve(A, b, v);
  }

}
