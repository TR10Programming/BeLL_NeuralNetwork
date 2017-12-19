package de.fk.neuralnetwork.math;

import de.fk.neuralnetwork.BasicNeuron;
import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.Neuron;
import de.fk.neuralnetwork.training.TrainingExample;
import java.util.Arrays;
import java.util.Random;
import static java.lang.Math.log;

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
     * Gewichte im Intervall [-0.5;0.5) enthält.
     *
     * @param length Länge des Arrays
     * @return Zufällige Gewichte
     */
    public static double[] generateRandomWeights(int length) {
        return rdm.doubles(length, -0.5, 0.5).toArray();
    }
    
    /**
     * Gibt den Fehler zwischen den erwarteten und den erhaltenen Werten zurück.
     *
     * @param actual Erhaltene Werte
     * @param expected Erwartete/Korrekte Werte
     * @return Fehlerrate
     */
    public static double getError(double[] actual, double[] expected) {
        if(actual.length != expected.length)
            throw new IllegalArgumentException("Die Längen der Arrays sind unterschiedlich.");
        double sum = 0.0;
        for(int i = 0; i < actual.length; i++)
            sum -= log(expected[i] == 1 ? actual[i] : (1 - actual[i]));
        return sum;
    }
    
    /**
     * Gibt den Fehler zwischen den erwarteten und den erhaltenen Werten zurück.
     * 
     * Bezieht die Regularization Rate Lambda mit ein.
     *
     * @param actual Erhaltene Werte
     * @param expected Erwartete/Korrekte Werte
     * @param lambda Regulrization Rate
     * @param nn Neural Network
     * @return Fehlerrate
     */
    public static double getRegularizedError(double[] actual, double[] expected, double lambda, NeuralNetwork nn) {
        double error = getError(actual, expected), lambdahalf = lambda / 2.0;
        //Regularisiere
        if(lambda != 0.0) {
            for(NeuralLayer l : nn.getLayers())
                for(Neuron n : l.getNeurons())
                    if(n instanceof BasicNeuron)
                        for(double weight : ((BasicNeuron) n).getWeights())
                            error += lambdahalf * weight * weight;
        }
        return error;
    }
    
    /**
     * Gibt die Fehler zwischen den erwarteten und den erhaltenen Werten jedes
     * einzelnen Neurons zurück.
     *
     * @param actual Erhaltene Werte
     * @param expected Erwartete/Korrekte Werte
     * @return Fehlerwerte aller Neuronen
     */
    public static double[] getErrors(double[] actual, double[] expected) {
        if(actual.length != expected.length)
            throw new IllegalArgumentException("Die Längen der Arrays sind unterschiedlich.");
        double[] errors = new double[actual.length];
        for(int i = 0; i < actual.length; i++)
            errors[i] = expected[i] - actual[i];
        return errors;
    }
    
    /**
     * Fügt dem übergebenen Array eine 1 vorne an und gibt ihn wieder zurück.
     *
     * @param vals
     * @return
     */
    public static double[] addBias(double[] vals) {
        double[] result = new double[vals.length + 1];
        System.arraycopy(vals, 0, result, 1, vals.length);
        result[0] = 1;
        return result;
    }
    
    /**
     * Fügt den Eingabewerten des übergebenen TrainingExamples eine 1 vorne an.
     *
     * @param te
     * @return
     */
    public static TrainingExample addBias(TrainingExample te) {
        double[] vals = te.getIn(), result = new double[vals.length + 1];
        System.arraycopy(vals, 0, result, 1, vals.length);
        result[0] = 1;
        te.setIn(result);
        return te;
    }
    
    
}
