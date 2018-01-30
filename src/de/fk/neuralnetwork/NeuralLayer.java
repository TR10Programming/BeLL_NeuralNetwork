package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *
 * @author Felix
 */
public class NeuralLayer implements Serializable {
    
    private static final long serialVersionUID = -7304433412748964606L/*-7456758553687520539L*/;
    
    private Neuron[] neurons;
    private ActivationFunction act;
    
    public NeuralLayer(int connectedNeurons, int neuronCount, boolean bias) {
        //System.out.println("Neues Layer generiert mit " + connectedNeurons + " Vorgängern und " + neuronCount + " Neuronen");
        neurons = new Neuron[neuronCount + (bias ? 1 : 0)];
        if(bias) neurons[0] = new BiasNeuron();
        for(int i = (bias ? 1 : 0); i < neurons.length; i++) neurons[i] = new BasicNeuron(connectedNeurons);
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
        return Arrays.stream(neurons).mapToDouble(n -> n.trigger(in, act)).toArray();
    }
    
    public double[] getErrorDeltas(double[] errors, double[] activationsBefore) {
        ArrayList<Double> errorDeltas = new ArrayList<>();
        for(int i = 0, j = 0; i < errors.length && j < neurons.length; i++, j++) {
            while(!(neurons[j] instanceof BasicNeuron) && j < neurons.length) j++;
            if(j >= neurons.length) break;
            errorDeltas.add(neurons[j].getErrorDelta(errors[i], act, activationsBefore));
        }
        return errorDeltas.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    public void calcAccumulatorMatrices(double[] errorDeltas, double[] activationsBefore, int threadId) {
        for(int i = 0, j = 0; i < errorDeltas.length && j < neurons.length; i++, j++) {
            while(!(neurons[j] instanceof BasicNeuron) && j < neurons.length) j++;
            if(j >= neurons.length) break;
            ((BasicNeuron) neurons[j]).calcAccumulatorMatrix(errorDeltas[i], activationsBefore, act, threadId);
        }
        //IntStream.range(0, neurons.length).filter(i -> neurons[i] instanceof BasicNeuron).forEach(i -> ((BasicNeuron) neurons[i]).calcAccumulatorMatrix(errors[i], activationsBefore, act, threadId));
    }
    
    public double[] getErrors(NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        return IntStream.range(0, neurons.length).filter(i -> neurons[i] instanceof BasicNeuron).mapToDouble(i -> neurons[i].getError(i, nextLayer, errorDeltasNextLayer)).toArray();
    }
    
    public void accumulate(double learningRate, double regularizationRate, double momentum) {
        for(Neuron n : neurons) if(n instanceof BasicNeuron) ((BasicNeuron) n).accumulate(learningRate, regularizationRate, momentum);
    }
    
    public void prepareForParallelBackprop(int threads) {
        for(Neuron n : neurons) if(n instanceof BasicNeuron) ((BasicNeuron) n).prepareForParallelBackprop(threads);
    }

}
