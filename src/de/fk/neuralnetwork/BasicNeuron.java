package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class BasicNeuron implements Neuron {

    private double activation, weightedInput, error;
    private double[] weights, accum;
    
    public BasicNeuron(int connectedNeurons) {
        weights = NeuralMath.generateRandomWeights(connectedNeurons);
        this.accum = Arrays.copyOf(weights, weights.length);
        //System.out.println("Neues Neuron mit " + connectedNeurons + " Vorgängern generiert. Gewichte: " + Arrays.toString(weights));
    }
    
    public BasicNeuron(double[] weights) {
        this.weights = weights;
        this.accum = Arrays.copyOf(weights, weights.length);
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
    @Override
    public double trigger(double[] input, ActivationFunction act) {
        return activation = act.apply(weightedInput = NeuralMath.applyWeights(input, weights));
    }

    @Override
    public double getWeightedInput() {
        return weightedInput;
    }

    @Override
    public double getActivation() {
        return activation;
    }

    @Override
    public void setError(double error) {
        this.error = error;
    }

    @Override
    public double getError() {
        return error;
    }
    
    public double[] getWeights() {
        return weights;
    }
    
    public double getWeightAt(int pos) {
        return weights[pos];
    }

    @Override
    public double getErrorDelta(ActivationFunction act) {
        return getError() * act.derivative(getWeightedInput());
    }
    
    public void resetAccumulatorMatrix() {
        this.accum = new double[weights.length];
    }
    
    public void calcAccumulatorMatrix(double[] activationsBefore, ActivationFunction act) {
        resetAccumulatorMatrix();
        double errorDelta = getErrorDelta(act);
        for(int i = 0; i < accum.length; i++) accum[i] += errorDelta * activationsBefore[i];
    }
    
    public void accumulate(double learningRate, double regularizationRate) {
        for(int i = 0; i < weights.length; i++)
            weights[i] += learningRate * (accum[i] - regularizationRate * weights[i]);
    }

    @Override
    public void calcError(int neuronPos, NeuralLayer nextLayer) {
        error = 0;
        double[] errorDeltasNextLayer = nextLayer.getErrorDeltas();
        Neuron[] neuronsNextLayer = nextLayer.getNeurons();
        for(int i = 0; i < neuronsNextLayer.length; i++)
            if(neuronsNextLayer[i] instanceof BasicNeuron)
                error += errorDeltasNextLayer[i] * ((BasicNeuron) neuronsNextLayer[i]).getWeightAt(neuronPos);
    }
    
}
