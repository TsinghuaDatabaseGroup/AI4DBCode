package edu.illinois.quicksel;

public interface SelectivityLearning extends SelectivityEstimator {

  public void addPermanentAssertion(Assertion a);

  public void addAssertion(Assertion a);

  public void assignOptimalWeights();

  public void reduceNumberOfAssertions(int target_number);

}
