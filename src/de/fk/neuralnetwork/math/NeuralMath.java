package de.fk.neuralnetwork.math;

import de.fk.neuralnetwork.BasicNeuron;
import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.Neuron;
import de.fk.neuralnetwork.training.TrainingExample;
import java.util.Random;
import static java.lang.Math.log;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 *
 * @author Felix
 */
public class NeuralMath {
    
    private static Random rdm  = new Random(/*System.currentTimeMillis()*/1081);

    /**
     * Multipliziert den Eingabearray mit den übergebenen Gewichten.
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
        double result = 0.0;
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
     * Gibt einen zweidimensionalen Array mit den spezifizierten Dimensionen
     * zurück, der zufällige Gewichte im Intervall [-0.5;0.5) enthält.
     *
     * @param length1 Länge der ersten Arraydimension
     * @param length2 Länge der zweiten Arraydimension
     * @return Zufällige Gewichte
     */
    public static double[][] generateRandomWeights(int length1, int length2) {
        return IntStream.range(0, length1)
                .parallel()
                .mapToObj(i -> rdm.doubles(length2, -0.5, 0.5).toArray())
                .toArray(double[][]::new);
    }
    
    /**
     * Gibt den Fehler zwischen den erwarteten und den erhaltenen Werten zurück.
     *
     * @param actual Erhaltene Werte
     * @param expected Erwartete/Korrekte Werte
     * @return Fehlerrate
     */
    public static double getError(double[] actual, double[] expected) {
        //System.out.println("Actual: " + Arrays.toString(actual) + " Expected: " + Arrays.toString(expected));
        if(actual.length != expected.length)
            throw new IllegalArgumentException("Die Längen der Arrays sind unterschiedlich.");
        double err, sum = 0.0;
        for(int i = 0; i < actual.length; i++) {
            err = -log(expected[i] == 1 ? actual[i] : (1 - actual[i]));
            sum += Double.isFinite(err) ? err : 999999;
        }
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
    
    /**
     * Gibt den zu erwartenden Output für ein neuronales Netz für ein 
     * Klassifizierungsproblem mit <code>classes</code> Klassen aus.
     * 
     * Dieser besteht aus einem Array der Länge <code>classes</code>
     * gefüllt mit Nullen und einer Eins an der Stelle <code>label</code>.
     *
     * @param label Label
     * @param classes Anzahl Klassen
     * @return Erwarteter Output
     */
    public static double[] getOutputForLabel(int label, int classes) {
        double[] output = new double[classes];
        output[label] = 1.0;
        return output;
    }
    
    /**
     * Gibt die bestimmte Klasse bzw den Index des höchsten Wertes im Array
     * zurück.
     *
     * @param out Ausgabe des Neuronalen Netzes
     * @return Bestimmte Klasse
     */
    public static int getPredictedLabel(double[] out) {
        double max = 0.0;
        int index = -1;
        for(int i = 0; i < out.length; i++) if(out[i] > max) {
            index = i;
            max = out[i];
        }
        return index;
    }
    
    /**
     * Wandelt den übergebenen zweidimensionalen Array (Matrix) in einen
     * eindimensionalen (Vektor) um, wobei alle Reihen der Matrix aneinander
     * gehangen werden.
     * 
     * Hinweis: Es muss garantiert sein, dass der zweidimensionale Array einer
     * Matrix der Größe m x n entspricht bzw. dass alle Arrays der zweiten
     * Dimension dieselbe Größe aufweisen.
     *
     * @param in Zweidimensionaler Array
     * @return Eindimensionaler Array
     */
    public static double[] flatten(double[][] in) {
        double[] flatData = new double[in.length * in[0].length];
        //for(int row = 0; row < in.length; row++) System.arraycopy(in[row], 0, flatData, row * in.length, in.length);
        IntStream.range(0, in.length).parallel().forEach(row -> System.arraycopy(in[row], 0, flatData, row * in.length, in.length));
        return flatData;
    }
    
    /**
     * Gibt eine exakte Kopie eines zweidimensionalen Double-Arrays zurück.
     *
     * @param matrix Zu klonender Array
     * @return Exakte Kopie (deep copy)
     */
    public static double[][] deepCopy(double[][] matrix) {
        return Arrays.stream(matrix).map(el -> el.clone()).toArray(double[][]::new);
    }
    
    /**
     * Führt eine Matrixmultiplikation mit einer Matrix und einem Vektor aus.
     * Dabei wird jede Zeile der Matrix mit den Elementen des Vektors
     * multipliziert und aufsummiert. Es entsteht ein Vektor mit der Länge der
     * Anzahl der Zeilen in der Ausgangsmatrix.
     *
     * @param mat
     * @param vec
     * @return Eingabevektor
     */
    public static double[] matmul(double[][] mat, double[] vec) {
        return Arrays.stream(mat)
                .mapToDouble(row -> 
                        IntStream.range(0, row.length)
                                .mapToDouble(col -> row[col] * vec[col])
                                .sum()
                ).toArray();
    }
    
    /**
     * Führt eine parallele Matrixmultiplikation mit einer Matrix und einem
     * Vektor aus. Dabei wird jede Zeile der Matrix mit den Elementen des
     * Vektors multipliziert und aufsummiert. Es entsteht ein Vektor mit der
     * Länge der Anzahl der Zeilen in der Ausgangsmatrix.
     *
     * @param mat
     * @param vec
     * @return Eingabevektor
     */
    public static double[] matmulParallel(double[][] mat, double[] vec) {
        return Arrays.stream(mat).parallel().mapToDouble(row -> 
                        IntStream.range(0, row.length)
                                .mapToDouble(col -> row[col] * vec[col])
                                .sum()
                ).toArray();
    }
    
    /**
     * Multipliziert die Gewichtsmatrix mit dem Eingabevektor und addiert den
     * Biasvektor. Zurückgegeben wird ein Vektor mit den Neuroneneingaben.
     *
     * @param weights
     * @param in
     * @param biases
     * @return Eingabevektor
     */
    public static double[] applyWeights(double[][] weights, double[] in, double[] biases) {
        return IntStream.range(0, weights[0].length)
                .mapToDouble(col ->
                        IntStream.range(0, weights.length)
                                .mapToDouble(row -> weights[row][col] * in[row])
                                .sum() + biases[col]
                ).toArray();
    }
    
    /**
     * Multipliziert die Gewichtsmatrix mit dem Eingabevektor und addiert den
     * Biasvektor. Nutzt parallele Streams. Zurückgegeben wird ein Vektor mit
     * den Neuroneneingaben.
     *
     * @param weights
     * @param in
     * @param biases
     * @return Eingabevektor
     */
    public static double[] applyWeightsParallel(double[][] weights, double[] in, double[] biases) {
        return IntStream.range(0, biases.length)
                .parallel()
                .mapToDouble(col ->
                        IntStream.range(0, weights.length)
                                .mapToDouble(row -> weights[row][col] * in[row])
                                .sum() + biases[col]
                ).toArray();
    }
    
    /**
     * Multipliziert die Gewichtsmatrix mit dem Eingabevektor, addiert den
     * Biasvektor und wendet die Aktivierungsfunktion an. Vektorwertige
     * Aktivierungsfunktionen wie Softmax dürfen nicht übergeben werden. Nutzt
     * parallele Streams. Zurückgegeben wird ein Vektor mit den Aktivierungen.
     *
     * @param weights
     * @param in
     * @param biases
     * @param act
     * @return Eingabevektor
     */
    public static double[] applyWeightsAndActivationFunctionParallel(double[][] weights, double[] in, double[] biases, ActivationFunction act) {
        return IntStream.range(0, biases.length)
                .parallel()
                .mapToDouble(col ->
                        act.apply(
                                IntStream.range(0, weights.length)
                                        .mapToDouble(row -> weights[row][col] * in[row])
                                        .sum() + biases[col]
                        )
                ).toArray();
    }
    
    /**
     * Gibt das Skalarprodukt der beiden Vektoren zurück. x^T*y bzw. (x) x (y)
     *
     * @param x
     * @param y
     * @return
     */
    public static double[] vecmul(double[] x, double[] y) {
        return IntStream.range(0, x.length)
                .mapToDouble(i -> x[i] * y[i])
                .toArray();
    }
    
    /**
     * Gibt das Skalarprodukt der beiden Vektoren zurück. x^T*y bzw. (x) x (y).
     * Nutzt bei ausreichender Vektorgröße parallele Streams.
     *
     * @param x
     * @param y
     * @return
     */
    public static double[] vecmulParallel(double[] x, double[] y) {
        IntStream stream = IntStream.range(0, x.length);
        if(x.length > 79) stream = stream.parallel();
        return stream.mapToDouble(i -> x[i] * y[i]).toArray();
    }
    
}
