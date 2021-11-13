package edu.illinois.quicksel.basic;
import javax.swing.*;

import edu.illinois.quicksel.heatmap.Gradient;
import edu.illinois.quicksel.heatmap.HeatMap;

import java.awt.*;

class HeatmapTest extends JFrame
{
    HeatMap panel;

    public HeatmapTest() throws Exception
    {
        super("Heat Map Frame");
        double[][] data = HeatMap.generateSinCosData(100);
        boolean useGraphicsYAxis = true;

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

    // this function will be run from the EDT

    private static void createAndShowGUI() throws Exception
    {
    	HeatmapTest hmf = new HeatmapTest();
        hmf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        hmf.setSize(500,500);
        hmf.setVisible(true);

    }

    public static void main(String[] args)
    {
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
}