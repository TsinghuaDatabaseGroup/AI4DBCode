package edu.illinois.quicksel;

public interface SelectivityEstimator {
  
  /**
   * 
   * @param query
   * @return selectivity ranging between 0.0 and 1.0 (inclusive both ends)
   */
  public double answer(Query query);

}
