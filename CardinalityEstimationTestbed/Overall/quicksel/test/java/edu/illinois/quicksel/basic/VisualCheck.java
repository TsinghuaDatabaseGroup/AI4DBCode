package edu.illinois.quicksel.basic;

import java.awt.Color;
import java.util.List;
import java.util.Vector;

import javax.swing.JFrame;
import javax.swing.SwingUtilities;

import edu.illinois.quicksel.quicksel.QuickSel;
import org.apache.commons.lang3.tuple.Pair;

import edu.illinois.quicksel.Assertion;
import edu.illinois.quicksel.Hyperrectangle;
import edu.illinois.quicksel.Query;
import edu.illinois.quicksel.isomer.Isomer;
import edu.illinois.quicksel.heatmap.Gradient;
import edu.illinois.quicksel.heatmap.HeatMap;


public class VisualCheck extends JFrame {

  HeatMap panel;

  public static void main(String[] args) {
    SwingUtilities.invokeLater(new Runnable()
    {
      public void run()
      {
        try
        {
          createAndShowGUI();
        }
        catch (Exception e)
        {
          System.err.println(e);
          e.printStackTrace();
        }
      }
    });
  }

  private static void createAndShowGUI() throws Exception
  {
    VisualCheck hmf = new VisualCheck();
    hmf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    hmf.setSize(500,500);
    hmf.setVisible(true);

  }

  public VisualCheck() throws Exception
  {
    super("Heat Map Frame");
    //        double[][] data = crumbsData();
    double[][] data = isomerData();
    boolean useGraphicsYAxis = false;

    // you can use a pre-defined gradient: 
    panel = new HeatMap(data, useGraphicsYAxis, Gradient.GRADIENT_BLUE_TO_RED);

    // or you can also make a custom gradient:

    Color[] gradientColors = new Color[]{Color.blue,Color.green,Color.yellow};
    Color[] customGradient = Gradient.createMultiGradient(gradientColors, 500);
    panel.updateGradient(customGradient);

    // set miscellaneous settings

    panel.setDrawLegend(true);

    panel.setTitle("Height (m)");
    panel.setDrawTitle(true);

    panel.setXAxisTitle("X-Distance (m)");
    panel.setDrawXAxisTitle(true);

    panel.setYAxisTitle("Y-Distance (m)");
    panel.setDrawYAxisTitle(true);

    panel.setCoordinateBounds(0, 6.28, 0, 6.28);

    panel.setDrawXTicks(true);
    panel.setDrawYTicks(true);

    this.getContentPane().add(panel);
  }


  public double[][] quickSelData() {
    int assertion_count = 100;

    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> permanent_assertions = SyntheticDataGenerator.twoDimPermanentAssertions(10, dataset);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(assertion_count, dataset);
    System.out.println("dataset and query set generations done.");

    // build Crumbs
    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = SyntheticDataGenerator.countSatisfyingItems(dataset, min_max_rec.toQuery()) / ((double) dataset.size());
    QuickSel quickSel = new QuickSel(min_max_rec, total_freq);

    for (Assertion a : permanent_assertions) {
      quickSel.addPermanentAssertion(a);
    }
    for (Assertion a : assertions) {
      quickSel.addAssertion(a);
    }
    boolean debug_output = true;
    quickSel.assignOptimalWeights(debug_output);


    // generate queryset
    int gridNum = 100;
    List<List<Query>> queryset = SyntheticDataGenerator.twoDimGaussianDatasetGridTestQueries(gridNum);


    // get answers
    double[][] data = new double[100][100];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[0].length; j++) {
        data[i][j] = quickSel.answer(queryset.get(i).get(j));
      }
    }

    return data;
  }


  public double[][] isomerData() {
    int assertion_count = 100;

    List<List<Double>> dataset = SyntheticDataGenerator.twoDimGaussianDataset(10000);
    List<Assertion> permanent_assertions = SyntheticDataGenerator.twoDimPermanentAssertions(10, dataset);
    List<Assertion> assertions = SyntheticDataGenerator.twoDimGaussianDatasetRandomAssertions(assertion_count, dataset);
    System.out.println("dataset and query set generations done.");

    // build Crumbs
    Vector<Pair<Double, Double>> min_max = new Vector<Pair<Double, Double>>();
    min_max.add(Pair.of(0.0, 1.0));
    min_max.add(Pair.of(0.0, 1.0));
    Hyperrectangle min_max_rec = new Hyperrectangle(min_max);
    double total_freq = SyntheticDataGenerator.countSatisfyingItems(dataset, min_max_rec.toQuery()) / ((double) dataset.size());
    int limit_node_count = 4000;
    Isomer isomer = new Isomer(min_max_rec, total_freq, limit_node_count);

    for (Assertion a : permanent_assertions) {
      isomer.addPermanentAssertion(a);
    }
    for (Assertion a : assertions) {
      isomer.addAssertion(a);
    }
    boolean debug_output = true;
    isomer.assignOptimalWeights(debug_output);


    // generate queryset
    int gridNum = 100;
    List<List<Query>> queryset = SyntheticDataGenerator.twoDimGaussianDatasetGridTestQueries(gridNum);


    // get answers
    double[][] data = new double[100][100];
    for (int i = 0; i < data.length; i++) {
      for (int j = 0; j < data[0].length; j++) {
        data[i][j] = isomer.answer(queryset.get(i).get(j));
      }
    }

    return data;
  }

}
