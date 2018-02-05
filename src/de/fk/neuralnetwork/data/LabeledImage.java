package de.fk.neuralnetwork.data;

/**
 * Repräsentiert ein Graustufenbild mit einem Label.
 *
 * @author Felix
 */
public class LabeledImage {

    private double[][] data;
    private int label;
    
    /**
     * Erstellt ein neues LabeledImage.
     *
     * @param data Daten
     * @param label Klasse/Label
     */
    public LabeledImage(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("== Label: ").append(label).append(" ==");
        for(int row = 0; row < data.length; row++) {
            for(int col = 0; col < data[row].length; col++) {
                double pixelVal = data[row][col];
                if(pixelVal == 0) sb.append(" ");
                else if(pixelVal < 1.0 / 3) sb.append(".");
                else if(pixelVal < 2.0 / 3) sb.append("x");
                else sb.append("X");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
}
