package edu.illinois.quicksel;

import org.jblas.DoubleMatrix;

public class Helpers {

  public static void printMatrix(DoubleMatrix A) {
    printMatrix(jblasMatrixToArray(A));
  }

  public static void printMatrix(double[][] A) {
    for (int i = 0; i < A.length; i++) {
      for (int j = 0; j < A[0].length; j++) {
        System.out.print(A[i][j] + " ");
      }
      System.out.print("\n");
    }
  }

  public static void printVector(double[] b) {
    for (int i = 0; i < b.length; i++) {
      System.out.print(b[i] + " ");
    }
    System.out.print("\n");
  }

  /**
   * Converts jblas.DoubleMatrix to a double-array representation.
   * @param A
   * @return
   */
  public static double[][] jblasMatrixToArray(DoubleMatrix A) {
    int m = A.rows;
    int n = A.columns;

    double[][] ret = new double[m][n]; 

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        ret[i][j] = A.get(i,j);
      }
    }

    return ret;
  }

  /**
   * Converts jblas.DoubleMatrix (containing a vector) to an array representation.
   * @param a
   * @return
   */
  public static double[] jblasVectorToArray(DoubleMatrix a) {
    int m = a.rows;
    int n = a.columns;
    assert(n == 1);

    double[] ret = new double[m];

    for (int i = 0; i < m; i++) {
      ret[i] = a.get(i);
    }

    return ret;
  }
}
