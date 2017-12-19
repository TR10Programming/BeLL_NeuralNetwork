package de.fk.neuralnetwork.data;

/**
 *
 * @author Felix
 */
public class LabeledImage {

    private int[][] data;
    private int label;
    
    public LabeledImage(int[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public int[][] getData() {
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
                int pixelVal = data[row][col];
                if(pixelVal == 0) sb.append(" ");
                else if(pixelVal < 256 / 3) sb.append(".");
                else if(pixelVal < 2 * (256 / 3)) sb.append("x");
                else sb.append("X");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
}
