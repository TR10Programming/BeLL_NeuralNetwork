package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;

/**
 *
 * @author Felix
 */
public class BasicNeuron implements Neuron {
    
    private static final long serialVersionUID = 4609093747635559427L/*8124483782436155978L*/;

    private double[] weights, weightsChange;
    private double[][] accum;
    
    public BasicNeuron(int connectedNeurons) {
        weights = NeuralMath.generateRandomWeights(connectedNeurons);
        this.accum = new double[1][weights.length];
        this.weightsChange = new double[weights.length];
        //System.out.println("Neues Neuron mit " + connectedNeurons + " Vorgängern generiert.");
    }
    
    public BasicNeuron(double[] weights) {
        this.weights = weights;
        this.accum = new double[1][weights.length];
        this.weightsChange = new double[weights.length];
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
        return act.apply(NeuralMath.applyWeights(input, weights));
    }
    
    public double[] getWeights() {
        return weights;
    }
    
    public double getWeightAt(int pos) {
        return weights[pos];
    }

    @Override
    public double getErrorDelta(double error, ActivationFunction act, double[] activationsBefore) {
        return error * act.derivative(NeuralMath.applyWeights(activationsBefore, weights));
    }
    
    public void resetAccumulatorMatrix(int threadId) {
        this.accum[threadId] = new double[weights.length];
    }
    
    public void calcAccumulatorMatrix(double errorDelta, double[] activationsBefore, ActivationFunction act, int threadId) {
        for(int i = 0; i < accum[threadId].length; i++) accum[threadId][i] += errorDelta * activationsBefore[i];
    }
    
    public void accumulate(double learningRate, double regularizationRate, double momentum) {//TODO Regularization
        for(int i = 0; i < weights.length; i++) {
            double weightsChangeBefore = weightsChange[i];
            //Werte aus allen Threads aufsummieren
            weightsChange[i] = 0.0;
            for(double[] acct : accum) {
                weightsChange[i] += acct[i];
            }
            if(weightsChangeBefore == 0.0) weights[i] += learningRate * weightsChange[i];
            else weights[i] += (1 - momentum) * (learningRate * weightsChange[i]) + momentum * weightsChangeBefore;
        }
        //Reset
        this.accum = new double[accum.length][weights.length];
    }

    @Override
    public double getError(int neuronPos, NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        double error = 0;
        Neuron[] neuronsNextLayer = nextLayer.getNeurons();
        for(int i = 0; i < neuronsNextLayer.length; i++)
            if(neuronsNextLayer[i] instanceof BasicNeuron)
                error += errorDeltasNextLayer[i] * ((BasicNeuron) neuronsNextLayer[i]).getWeightAt(neuronPos);
        return error;
    }
    
    public void prepareForParallelBackprop(int threads) {
        this.accum = new double[threads][weights.length];
    }
    
}
