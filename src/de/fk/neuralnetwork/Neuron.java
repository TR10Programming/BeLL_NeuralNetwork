package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class Neuron {

    private double[] weights;
    
    public Neuron(int connectedNeurons) {
        weights = NeuralMath.generateRandomWeights(connectedNeurons);
        System.out.println("Neues Neuron mit " + connectedNeurons + " Vorgängern generiert. Gewichte: " + Arrays.toString(weights));
    }
    
    public Neuron(double[] weights) {
        this.weights = weights;
    }
    
    /**
     * Berechnet den Ausgabewert des Neurons anhand der Gewichte des Neurons,
     * dem gegebenen Eingabevektor und der Aktivierungsfunktion.
     *
     * @param input Eingabevektor
     * @param act Zu verwendende Aktivierungsfunktion
     * @return Ausgabewert
     * @see NeuralMath#applyWeights(double[], double[]) 
     */
    public double trigger(double[] input, ActivationFunction act) {
        return act.apply(NeuralMath.applyWeights(input, weights));
    }
    
}
