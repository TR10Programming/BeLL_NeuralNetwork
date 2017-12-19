package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class NeuralLayer {
    
    private Neuron[] neurons;
    private ActivationFunction act;
    
    public NeuralLayer(int connectedNeurons, int neuronCount, boolean bias) {
        //System.out.println("Neues Layer generiert mit " + connectedNeurons + " Vorgängern und " + neuronCount + " Neuronen");
        neurons = new Neuron[neuronCount + (bias ? 1 : 0)];
        for(int i = 0; i < neuronCount; i++) neurons[i] = new BasicNeuron(connectedNeurons);
        if(bias) neurons[neuronCount] = new BiasNeuron();
        act = ActivationFunction.DEFAULT_ACTIVATION_FUNCTION;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public void setActivationFunction(ActivationFunction act) {
        this.act = act;
    }

    public ActivationFunction getActivationFunction() {
        return act;
    }
    
    public double[] trigger(double[] in) {
        return Arrays.stream(neurons).parallel().mapToDouble(n -> n.trigger(in, act)).toArray();
    }
    
    public void setErrors(double[] errors) {
        if(neurons.length != errors.length) throw new IllegalArgumentException("Es sind " + neurons.length + " Neuronen vorhanden, aber " + errors.length + " Fehlerwerte wurden übergeben.");
        for(int i = 0; i < neurons.length; i++) neurons[i].setError(errors[i]);
    }
    
    public double[] getErrorDeltas() {
        return Arrays.stream(neurons).parallel().mapToDouble(n -> n.getErrorDelta(act)).toArray();
    }
    
    public double getActivationAt(int pos) {
        return neurons[pos].getActivation();
    }
    
    public double[] getActivations() {
        return Arrays.stream(neurons).parallel().mapToDouble(n -> n.getActivation()).toArray();
    }
    
    public void calcAccumulatorMatrices(double[] activationsBefore) {
        for(Neuron n : neurons) if(n instanceof BasicNeuron) ((BasicNeuron) n).calcAccumulatorMatrix(activationsBefore, act);
    }
    
    public void calcErrors(NeuralLayer nextLayer) {
        for(int i = 0; i < neurons.length; i++) {
            neurons[i].calcError(i, nextLayer);
        }
    }
    
    public void accumulate(double learningRate, double regularizationRate) {
        for(Neuron n : neurons) if(n instanceof BasicNeuron) ((BasicNeuron) n).accumulate(learningRate, regularizationRate);
    }

}
