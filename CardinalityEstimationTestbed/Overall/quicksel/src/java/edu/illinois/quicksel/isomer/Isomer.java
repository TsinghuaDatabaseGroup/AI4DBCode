package edu.illinois.quicksel.isomer;

import edu.illinois.quicksel.OptimizerWrapper;
import edu.illinois.quicksel.*;
import edu.illinois.quicksel.quicksel.ProjectionAlgorithm;

import java.util.ArrayList;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;

import org.apache.commons.lang3.tuple.Triple;
import org.jblas.DoubleMatrix;

import static java.lang.Math.*;


/**
 * Our implementation of the ISOMER algorithm.
 */
public class Isomer implements SelectivityLearning {

  private IsomerNode root;

  private Hyperrectangle min_max;

  private int variableCnt = 0;

  private double total_freq;

  private Vector<Assertion> permanent_assertions = new Vector<Assertion>();

  private Vector<Assertion> assertions = new Vector<Assertion>();

  private int limit_node_count = -1;
  
  private boolean isSTHoles = false;
  
  private double[] qp_node_weights;

  public Isomer(Hyperrectangle min_max_range) {
    this(min_max_range, 1.0, -1);
  }

  public Isomer(Hyperrectangle min_max_range, int limit_node_count) {
    this(min_max_range, 1.0, limit_node_count);
  }

  public Isomer(Hyperrectangle min_max_range, double total_freq) {
    this(min_max_range, total_freq, -1);
  }

  public Isomer(Hyperrectangle min_max_range, double total_freq, int limit_node_count) {
    root = new IsomerNode(min_max_range, total_freq);
    min_max = min_max_range;
    this.total_freq = total_freq;
//    addPermanentAssertion(new Assertion(min_max_range.toQuery(), total_freq));
    this.limit_node_count = limit_node_count;
  }

  public int countNodes() {
    return IsomerNodeCounter.count(root);
  }
  
  public void activateSTHoles() {
    isSTHoles = true;
    limit_node_count = 4000;
  }

  public String toString() {
    String ret = root.toString();
    ret += String.format("\nTotal number of nodes: %d\n", countNodes());
    return ret;
  }

  private Hyperrectangle rectangleFromQuery(Query query) {
    return query.rectangleFromQuery(this.min_max);
  }

  @Override
  public void addAssertion(Assertion a) {
    root.add(new IsomerNode(rectangleFromQuery(a.query), a.freq));
    assertions.add(a);

    if (isSTHoles && limit_node_count > 0) {
      reduceNumberOfNodes(limit_node_count);
    }
  }

  @Override
  public void addPermanentAssertion(Assertion a) {
    root.add(new IsomerNode(rectangleFromQuery(a.query), a.freq));
    permanent_assertions.add(a);
  }

  @Override
  public void assignOptimalWeights() {
    assignOptimalWeights(false);
  }
  
  public void assignOptimalWeightsQP(boolean debug_output) {
    Vector<IsomerNode> nodes = IsomerNodeExtractor.extractNodes(root);

    // collect all assertions for convenience.
    Vector<Assertion> all_assertions = new Vector<Assertion>();
    all_assertions.addAll(permanent_assertions);
    all_assertions.addAll(assertions);
    
    int n = all_assertions.size();
    int m = nodes.size();
    variableCnt = nodes.size();
    
    Vector<Hyperrectangle> qrec = new Vector<Hyperrectangle>();
    for (int i = 0; i < n; i++) {
      qrec.add(rectangleFromQuery(all_assertions.get(i).query));
    }

    DoubleMatrix P = DoubleMatrix.zeros(m, m);
    DoubleMatrix A = DoubleMatrix.zeros(n, m);
    double[] b = new double[n];
    double[] v = new double[m];
    
    for (int i = 0; i < m; i++) {
      double vol = nodes.get(i).exclusiveVol();
      P.put(i, i, 1.0 / vol);
    }
    P.addi(DoubleMatrix.eye(m).muli(1e-3));
    
    for (int i = 0; i < qrec.size(); i++) {
      for (int j = 0; j < nodes.size(); j++) {
        A.put(i, j, nodes.get(j).exclusiveIntersect(qrec.get(i)) / nodes.get(j).exclusiveVol());
      }
    }

    for (int i = 0; i < all_assertions.size(); i++) {
      b[i] = all_assertions.get(i).freq;
    }

    for (int i = 0; i < nodes.size(); i++) {
      v[i] = nodes.get(i).exclusiveVol();
    }
    
    
    // Now, solve
    double constraints_weights = 1e6;
    double[] sol = ProjectionAlgorithm.solve(P, A, new DoubleMatrix(b), constraints_weights, 0);
    OptimizerWrapper.constraint_penalty = constraints_weights;
//    double[] sol = OptimizerWrapper.safe_squared_norm_solver(P, A, new DoubleMatrix(b));
    qp_node_weights = sol;
    
    if (debug_output) {
      DoubleMatrix x = new DoubleMatrix(sol);
      DoubleMatrix gap = A.mmul(x).sub(new DoubleMatrix(b));
      System.out.println("weights sum: " + x.sum());
      System.out.println("l1 gap: " + gap.norm1());
    }

    // update weights
    Vector<Double> weights = new Vector<Double>();
    for (int i = 0; i < sol.length; i++) {
      weights.add(sol[i]);
    }
    IsomerNodeWeightSetter.setWeights(root, weights);
  }
  
  public double answerQP(Query query) {
    Vector<IsomerNode> nodes = IsomerNodeExtractor.extractNodes(root);
    
    double sum_freq = 0.0;
    for (int i = 0; i < nodes.size(); i++) {
      IsomerNode node = nodes.get(i);
      sum_freq += node.exclusiveIntersect(rectangleFromQuery(query)) / node.exclusiveVol() * qp_node_weights[i];
    }
    
    return sum_freq;
  }

  public void assignOptimalWeights(boolean debug_output) {
    Vector<IsomerNode> nodes = IsomerNodeExtractor.extractNodes(root);

    // collect all assertions for convenience.
    Vector<Assertion> all_assertions = new Vector<Assertion>();
    all_assertions.addAll(permanent_assertions);
    all_assertions.addAll(assertions);

    Vector<Hyperrectangle> qrec = new Vector<Hyperrectangle>();
    for (int i = 0; i < all_assertions.size(); i++) {
      qrec.add(rectangleFromQuery(all_assertions.get(i).query));
    }

    int n = all_assertions.size();
    int m = nodes.size();
    variableCnt = nodes.size();

    double[][] A = new double[n][m];
    double[] b = new double[n];
    double[] v = new double[m];

    // optimization constraints
    for (int i = 0; i < qrec.size(); i++) {
      for (int j = 0; j < nodes.size(); j++) {
        A[i][j] = nodes.get(j).exclusiveIntersect(qrec.get(i)) / nodes.get(j).exclusiveVol();
      }
    }

    for (int i = 0; i < all_assertions.size(); i++) {
      b[i] = all_assertions.get(i).freq;
    }

    for (int i = 0; i < nodes.size(); i++) {
      v[i] = nodes.get(i).exclusiveVol();
    }


    // optimize
    double[] sol;
    //		double[] sol = QPSolver.solve(P, q, G, h, A, b);
    //		double[] sol = OptimizerWrapper.safe_squared_norm_solver(A, b, v);
    String method = "entropy";
    if (method.equals("entropy")) {
      sol = OptimizerWrapper.entropy_solver(A, b, v);
    } else {
      sol = new double[10];
    }

    if (debug_output) {
      DoubleMatrix x = new DoubleMatrix(sol);
      DoubleMatrix A1 = new DoubleMatrix(A);
      DoubleMatrix gap = A1.mmul(x).sub(new DoubleMatrix(b));
      System.out.println("weights sum: " + x.sum());
      System.out.println("l1 gap: " + gap.norm1());
    }

    // update weights
    Vector<Double> weights = new Vector<Double>();
    for (int i = 0; i < sol.length; i++) {
      weights.add(sol[i]);
    }
    IsomerNodeWeightSetter.setWeights(root, weights);
  }

  public void reduceNumberOfNodes(int target_number) {
    assert(target_number > 0);
    if (this.countNodes() <= target_number) return;

    List<Triple<IsomerNode, IsomerNode, Double>> penalties = IsomerPaneltyCalculator.calculatePenalties(root);

    while (this.countNodes() > target_number) {
      Triple<IsomerNode, IsomerNode, Double> min_penalty_entry = penalties.get(0);
      mergeAndUpdatePenalty(min_penalty_entry.getLeft(), min_penalty_entry.getMiddle(), penalties);
    }
  }

  @Override
  public void reduceNumberOfAssertions(int target_number) {
    target_number -= permanent_assertions.size();
    Vector<Assertion> reduced_assertions = new Vector<Assertion>();
    reduced_assertions.add(assertions.get(0));
    reduced_assertions.addAll(assertions.subList(max(1, assertions.size() - target_number), assertions.size()));
    assertions = reduced_assertions;
  }

  private void mergeAndUpdatePenalty(IsomerNode parent, IsomerNode child, List<Triple<IsomerNode, IsomerNode, Double>> penalties) {
    List<IsomerNode> grandchildren = child.children;

    // 1. merge parent and child
    parent.mergeWithChlid(child);

    // 2. remove the entries from 'penalties' that include 'child' as their first elements.
    int i = 0;
    while (i < penalties.size()) {
      if (penalties.get(i).getLeft().equals(child)) {
        penalties.remove(i);	// this part is not optimally fast because the underlying implementation of 'penalties' is currently ArrayList.
      }
      else {
        i += 1;
      }
    }

    // 3. remove the entries from 'penalties' that include 'child' as their second elements.
    i = 0;
    while (i < penalties.size()) {
      if (penalties.get(i).getMiddle().equals(child)) {
        penalties.remove(i);	// this part is not optimally fast because the underlying implementation of 'penalties' is currently ArrayList.
      }
      else {
        i += 1;
      }
    }

    // 4. add the entries to 'penalties'; they should include 'parent' as the first elements and the nodes in 'grandchildren' as
    //    the second elements.
    for (IsomerNode g : grandchildren) {
      penalties.add(Triple.of(parent, g, parent.penalty(g)));
    }
    PenaltyUtils.sortPenalties(penalties);
  }

  @Override
  public double answer(Query query) {
    return IsomerAnswerCalculator.computeAnswer(root, query.rectangleFromQuery(min_max));
  }

  public int getVariableCnt() {
    return variableCnt;
  }
}


class IsomerAnswerCalculator implements Visitor {

  private double weights_sum = 0;

  private Hyperrectangle query_rec;

  private IsomerAnswerCalculator(Hyperrectangle query_rec) {
    this.query_rec = query_rec;
  }

  @Override
  public void visit(IsomerNode node) {
    //		System.out.println(node.exclusiveIntersect(query_rec) / node.exclusiveVol());
    weights_sum += node.exclusiveIntersect(query_rec) / node.exclusiveVol() * node.weight;
  }

  public static double computeAnswer(IsomerNode root, Hyperrectangle query_rec) {
    IsomerAnswerCalculator c = new IsomerAnswerCalculator(query_rec);
    root.accept(c);
    return c.weights_sum;
  }
}


class PenaltyUtils {

  public static List<Triple<IsomerNode, IsomerNode, Double>> sortPenalties(List<Triple<IsomerNode, IsomerNode, Double>> penalties) {
    Collections.sort(penalties, new Comparator<Triple<IsomerNode, IsomerNode, Double>>() {
      public int compare(Triple<IsomerNode, IsomerNode, Double> a, Triple<IsomerNode, IsomerNode, Double> b) {
        return a.getRight().compareTo(b.getRight());
      }
    });

    return penalties;
  }

  public static void printPenalties(List<Triple<IsomerNode, IsomerNode, Double>> penalties) {
    for (Triple<IsomerNode, IsomerNode, Double> t : penalties) {
      System.out.println(String.format("(%d, %d, %f)",
          t.getLeft().hashCode(), t.getMiddle().hashCode(), t.getRight()));
    }
    System.out.println();
  }
}


class IsomerPaneltyCalculator implements Visitor {

  private List<Triple<IsomerNode, IsomerNode, Double>> node_triples
  = new ArrayList<Triple<IsomerNode, IsomerNode, Double>>();

  private IsomerPaneltyCalculator() {};

  @Override
  public void visit(IsomerNode node) {
    for (IsomerNode c : node.children) {
      node_triples.add(Triple.of(node, c, node.penalty(c)));
    }
  }

  protected static List<Triple<IsomerNode, IsomerNode, Double>> calculatePenalties(IsomerNode root) {
    IsomerPaneltyCalculator e = new IsomerPaneltyCalculator();
    root.accept(e);
    return PenaltyUtils.sortPenalties(e.node_triples);
  }
}


class IsomerNodeExtractor implements Visitor {

  private Vector<IsomerNode> nodes = new Vector<IsomerNode>();

  private IsomerNodeExtractor() {}

  @Override
  public void visit(IsomerNode node) {
    nodes.add(node);
  }

  protected static Vector<IsomerNode> extractNodes(IsomerNode root) {
    IsomerNodeExtractor e = new IsomerNodeExtractor();
    root.accept(e);
    return e.nodes;
  }
}


class IsomerNodeWeightSetter implements Visitor {

  private Vector<Double> weights;

  private int idx;

  private IsomerNodeWeightSetter(Vector<Double> weights) {
    this.weights = weights;
    idx = 0;
  }

  @Override
  public void visit(IsomerNode node) {
    node.weight = weights.get(idx);
    idx += 1;
  }

  protected static void setWeights(IsomerNode root, Vector<Double> weights) {
    IsomerNodeWeightSetter s = new IsomerNodeWeightSetter(weights);
    root.accept(s);
  }
}


class IsomerNodeCounter implements Visitor {

  private int counter;

  private IsomerNodeCounter() {
    counter = 0;
  }

  public void visit(IsomerNode node) {
    counter += 1;
  }

  protected static int count(IsomerNode root) {
    IsomerNodeCounter c = new IsomerNodeCounter();
    root.accept(c);
    return c.counter;
  }
}

class IsomerNodeWeightsSum implements Visitor {

  private double weights = 0;

  public void visit(IsomerNode node) {
    weights += node.weight;
  }

  public static double calculate(IsomerNode root) {
    IsomerNodeWeightsSum s = new IsomerNodeWeightsSum();
    root.accept(s);
    return s.weights;
  }
}
