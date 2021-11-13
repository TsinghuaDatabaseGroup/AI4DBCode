package edu.illinois.quicksel.isomer;

import static java.lang.Math.abs;
import static java.lang.Math.max;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Vector;

import edu.illinois.quicksel.Traversable;
import org.apache.commons.lang3.tuple.Pair;

import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Visitor;


/**
 * Underlying node for ISOMER.
 * 
 * @author Yongjoo Park
 */
public class IsomerNode implements Traversable {

  protected Hyperrectangle range;

  protected List<IsomerNode> children = new ArrayList<>();

  protected double weight;

  public IsomerNode(Hyperrectangle range) {
    this(range, 0);
  }

  public IsomerNode(Hyperrectangle range, double weight) {
    this.range = range;
    this.weight = weight;
  }

  public String toString() {
    return toStringRoot(0);
  }

  public String toStringRoot(int indent) {
    String ret = String.join("",  Collections.nCopies(indent, " "));
    //		ret += "{" + this.hashCode() + "}  ";
    ret += range.toString() + String.format("  (%s)", String.valueOf(weight));
    ret += String.format("  total vol: %.4f,\texclusive vol: %.4f)\n", this.totalVol(), this.exclusiveVol());
    for (IsomerNode c : children) {
      ret += c.toStringRoot(indent + 2);
    }
    return ret;
  }

  protected double totalWeight() {
    double myweight = 0;

    for (IsomerNode c : children) {
      myweight += c.totalWeight();
    }

    return myweight + this.weight;
  }

  protected double totalVol() {
    return this.range.vol();
  }

  protected double exclusiveVol() {
    double cvol = 0.0;
    for (IsomerNode c : children) {
      cvol += c.totalVol();
    }

    return max(1e-15, this.totalVol() - cvol);
  }

  protected double exclusiveIntersect(Hyperrectangle rec) {
    double vol = this.intersect(rec);

    for (IsomerNode c : children) {
      vol -= c.intersect(rec);
    }

    return vol;
  }

  protected double intersect(IsomerNode other) {
    return intersect(other.range);
  }

  protected double intersect(Hyperrectangle rec) {
    return this.range.intersect(rec);
  }

  /**
   * I first introduced this function to prevent the cases in which the exclusive volume
   * of this (or parent) node is too small (or even negative for some computational errors);
   * however, I don't think such cases happen very frequently.
   * @return
   */
  protected Vector<IsomerNode> trim() {		
    double cvol = 0;
    for (IsomerNode c : children) {
      cvol += c.totalVol();
    }

    if (cvol >= this.totalVol() - 1e-15) {
      // the sum of children volumes is bigger (exception case)
      Vector<IsomerNode> new_nodes = new Vector<IsomerNode>();
      for (IsomerNode c : children) {
        new_nodes.addAll(c.trim());
      }
      return new_nodes;
    }
    else {
      Vector<IsomerNode> new_children = new Vector<IsomerNode>();
      for (IsomerNode c : children) {
        new_children.addAll(c.trim());
      }
      this.children = new_children;

      Vector<IsomerNode> new_nodes = new Vector<IsomerNode>();
      new_nodes.add(this);
      return new_nodes;
    }
  }

  private Pair<Optional<Hyperrectangle>, Optional<Hyperrectangle>>
  single_div_rec(Hyperrectangle rec, int k, double v) {
    Optional<Hyperrectangle> left = Optional.empty();
    Optional<Hyperrectangle> right = Optional.empty();
    Pair<Double, Double> kintvl = rec.intervals.get(k);

    if (kintvl.getRight() <= v) left = Optional.of(new Hyperrectangle(rec));
    else if (kintvl.getLeft() >= v) right = Optional.of(new Hyperrectangle(rec));
    else {
      left = Optional.of(new Hyperrectangle(rec));
      right = Optional.of(new Hyperrectangle(rec));
      left.get().intervals.set(k, Pair.of(kintvl.getLeft(), v));
      right.get().intervals.set(k,  Pair.of(v, kintvl.getRight()));
    }

    return Pair.of(left, right);
  }

  private Pair<Optional<IsomerNode>, Optional<IsomerNode>>
  divide(int k, double v) {
    Pair<Optional<Hyperrectangle>, Optional<Hyperrectangle>> lr = single_div_rec(this.range, k, v);

    Optional<IsomerNode> myleft = lr.getLeft().map(x -> new IsomerNode(x));
    Optional<IsomerNode> myright = lr.getRight().map(x -> new IsomerNode(x));

    if (!myleft.isPresent()) myright.get().weight = this.weight;
    else if (!myright.isPresent()) myleft.get().weight = this.weight;
    else {
      double vol_sum = myleft.get().totalVol() + myright.get().totalVol();
      myleft.get().weight = this.weight * myleft.get().totalVol() / vol_sum;
      myright.get().weight = this.weight * myright.get().totalVol() / vol_sum;
    }


    for (IsomerNode c : this.children) {
      Pair<Optional<IsomerNode>, Optional<IsomerNode>> clr = c.divide(k, v);
      if (clr.getLeft().isPresent()) myleft.get().children.add(clr.getLeft().get());
      if (clr.getRight().isPresent()) myright.get().children.add(clr.getRight().get());
    }

    return Pair.of(myleft, myright);
  }

  /**
   * 'this' node cracks the 'other' node
   * @param other
   * @return A pair of the nodes that will be added as children of 'this' node and the nodes that should
   * be added to the children of the parent of the 'other' node. In other words, the nodes in the second
   * pair replaces the 'other' node.
   */
  private Pair<Vector<IsomerNode>, Vector<IsomerNode>>
  crack(IsomerNode other) {

    if (this.intersect(other) == 0) {
      Vector<IsomerNode> a = new Vector<IsomerNode>();
      a.add(other);
      return Pair.of(new Vector<IsomerNode>(), a);
    }

    Vector<IsomerNode> otherpieces = new Vector<IsomerNode>();
    IsomerNode remaining = other;

    for (int k = 0; k < this.range.d; k++) {
      if (this.intersect(remaining) == 0) break;

      Pair<Double, Double> intvl = this.range.intervals.get(k);
      Pair<Optional<IsomerNode>, Optional<IsomerNode>> lr = remaining.divide(k, intvl.getLeft());

      if (lr.getLeft().isPresent()) {
        otherpieces.addAll(lr.getLeft().get().trim());
      }

      lr = lr.getRight().get().divide(k, intvl.getRight());
      if (lr.getRight().isPresent()) otherpieces.addAll(lr.getRight().get().trim());

      remaining = lr.getLeft().get();
    }

    return Pair.of(remaining.trim(), otherpieces); 
  }

  private boolean doesBelongTo(IsomerNode me, IsomerNode other) {
    return me.totalVol() == me.intersect(other);
  }

  private boolean doesInclude(IsomerNode me, IsomerNode other) {
    return other.totalVol() == me.intersect(other); 
  }

  private boolean isEqualNode(IsomerNode me, IsomerNode other) {
    return doesBelongTo(me, other) && doesInclude(me, other);
  }

  protected void add(IsomerNode node) {

    // check if there is any identical node
    for (IsomerNode c : this.children) {
      if (isEqualNode(c, node)) return;
    }

    // check if there is any child that includes 'node'
    for (IsomerNode c : this.children) {
      if (doesInclude(c, node)) {
        c.add(node);
        return;
      }
    }

    double old_total_weight = this.totalWeight();

    // now, 'node' must be added to self.children
    Vector<IsomerNode> left_children = new Vector<IsomerNode>();
    Vector<IsomerNode> node_children = new Vector<IsomerNode>();

    // check if there is any child that 'node' includes
    for (int i = 0; i < this.children.size(); i++) {
      IsomerNode c = this.children.get(i);
      if (doesInclude(node, c)) node_children.add(c);
      else left_children.add(c);
    }

    Vector<IsomerNode> new_children = new Vector<IsomerNode>();
    for (IsomerNode c : left_children) {
      Pair<Vector<IsomerNode>, Vector<IsomerNode>> my_other = node.crack(c);
      if (my_other.getLeft().size() > 0) node_children.addAll(my_other.getLeft());
      new_children.addAll(my_other.getRight());
    }

    node.children = node_children;
    for (IsomerNode c : node.children) {
      node.weight -= c.totalWeight();
    }
    node.weight = max(0, node.weight);

    this.children = new_children;
    this.children.addAll(node.trim());

    for (IsomerNode c : children) {
      old_total_weight -= c.totalWeight();
    }
    this.weight = max(0, old_total_weight);
  }

  @Override
  public void accept(Visitor v) {
    v.visit(this);
    for (IsomerNode c : this.children) {
      c.accept(v);
    }
  }

  protected double penalty(IsomerNode child) {
    double my_vol = this.exclusiveVol();
    double child_vol = child.exclusiveVol();
    double combined_vol = my_vol + child_vol;

    double parent_penalty = abs(this.weight - (this.weight + child.weight) * my_vol / combined_vol);
    double child_penalty = abs(child.weight - (this.weight + child.weight) * child_vol / combined_vol);

    return parent_penalty + child_penalty;
  }

  protected void mergeWithChlid(IsomerNode child) {
    // make sure child is one of this node's child.
    boolean doesExist = false;
    for (IsomerNode c : children) {
      if (c == child) {
        doesExist = true;
        break;
      }
    }
    if (!doesExist) {
      assert(false);
    }

    this.weight += child.weight;
    children.remove(child);
    this.children.addAll(child.children);
  }

}