package de.fk.neuralnetwork.math;

import java.util.Random;

/**
 *
 * @author Felix
 */
public class NeuralMath {
    
    private static Random rdm  = new Random(1081);

    /**
     * Errechnet das Skalarprodukt des Eingabevektors und des Gewichtevektors.
     * 
     * Das Gewichtearray darf größer sein als das Eingabearray, in diesem Fall 
     * werden die restlichen Gewichte zum Skalarprodukt addiert (Bias). Das
     * Eingabearray darf nicht größer sein als das Gewichtearray.
     *
     * @param in
     * @param weights
     * @return Skalarprodukt der beiden Vektoren
     * @throws ArrayIndexOutOfBoundsException wenn der Eingabearray größer als der Gewichtearray ist
     */
    public static double applyWeights(double[] in, double[] weights) throws ArrayIndexOutOfBoundsException {
        double result = 0f;
        for(int i = 0; i < in.length; i++) result += in[i] * weights[i];
        for(int w = in.length; w < weights.length; w++) result += weights[w];
        return result;
    }
    
    /**
     * Gibt einen Array mit der spezifizierten Länge zurück, der zufällige
     * Gewichte im Intervall [0;1) enthält.
     *
     * @param length Länge des Arrays
     * @return Zufällige Gewichte
     */
    public static double[] generateRandomWeights(int length) {
        return rdm.doubles(length, -0.5, 0.5).toArray();
    }
    
}
