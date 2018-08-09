package de.fk.neuralnetwork.nets;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 *
 * @author Felix
 */
@Deprecated
public class NeuralLayer {

    private double[][] weights, gradients;
    private double[] biases, inputs, activations, deltas, biasgradients;
    private ActivationFunction act;
    
    public NeuralLayer(int connectedNeurons, int neuronCount, ActivationFunction act) {
        weights = NeuralMath.generateRandomWeights(connectedNeurons, neuronCount);
        biases = new double[neuronCount];
        gradients = new double[connectedNeurons][neuronCount];
        inputs = new double[neuronCount];
        activations = new double[neuronCount];
        deltas = new double[neuronCount];
        biasgradients = new double[neuronCount];
        this.act = act;
    }
    
    public NeuralLayer(double[][] weights, double[] biases, ActivationFunction act) {
        this.weights = weights;
        this.biases = biases;
        gradients = new double[weights.length][biases.length];
        inputs = new double[biases.length];
        activations = new double[biases.length];
        deltas = new double[biases.length];
        biasgradients = new double[biases.length];
        this.act = act;
    }

    public void setActivationFunction(ActivationFunction act) {
        this.act = act;
    }

    public ActivationFunction getActivationFunction() {
        return act;
    }
    
    public int getNeuronCount() {
        return biases.length;
    }
    
    public double[] trigger(double[] in) {
        //z=Wx+b inputs=weights*in+biases
        inputs = NeuralMath.applyWeights(weights, in, biases);
        return activations = act.applyAll(inputs);
    }
    
    public double[] triggerParallel(double[] in) {
        //z=Wx+b inputs=weights*in+biases
        inputs = NeuralMath.applyWeightsParallel(weights, in, biases);
        return activations = act.applyAll(inputs);
    }
    
    public double[] getOutput(double[] in) {
        if(act.needsAllInputs()) return act.applyAll(NeuralMath.applyWeightsParallel(weights, in, biases));
        else return NeuralMath.applyWeightsAndActivationFunctionParallel(weights, in, biases, act);
    }

    public double[] getActivations() {
        return activations;
    }
    
    /**
     * Berechnet die Delta-Werte für diese Schicht und anschließend die
     * Gradienten für jedes Gewicht.
     *
     * @param errorsFromNextLayer
     * @param activationsFromLayerBefore
     */
    public void backprop(double[] errorsFromNextLayer, double[] activationsFromLayerBefore) {
        //Berechne Deltas für Neuronen delta(i)=error(i+1) .* g'(z(i))
        deltas = NeuralMath.vecmul(errorsFromNextLayer, act.derivativeAll(inputs));
        //Berechne Gradienten/Accumulator Matrix und addiere diesen
        for(int conneuron = 0; conneuron < weights.length; conneuron++)
            for(int neuron = 0; neuron < weights[conneuron].length; neuron++)
                gradients[conneuron][neuron] -= activationsFromLayerBefore[conneuron] * deltas[neuron];
        //Gradienten für Biases = - Deltas
        for(int i = 0; i < biases.length; i++) biasgradients[i] -= deltas[i];
    }
    
    /**
     * Berechnet die Delta-Werte für diese Schicht und anschließend parallel die
     * Gradienten für jedes Gewicht. Nutzt bei ausreichender Schichtgröße
     * parallele Streams.
     *
     * @param errorsFromNextLayer
     * @param activationsFromLayerBefore
     */
    public void backpropParallel(double[] errorsFromNextLayer, double[] activationsFromLayerBefore) {
        //Berechne Deltas für Neuronen
        deltas = NeuralMath.vecmulParallel(errorsFromNextLayer, act.derivativeAll(inputs));
        //Berechne Gradienten/Accumulator Matrix und addiere diesen
        IntStream.range(0, biases.length).parallel().forEach(neuron -> {
                        for(int conneuron = 0; conneuron < weights.length; conneuron++)
                            gradients[conneuron][neuron] -= activationsFromLayerBefore[conneuron] * deltas[neuron];
                        biasgradients[neuron] -= deltas[neuron];
                });
    }
    
    public double[] getErrors() {
        return NeuralMath.matmul(weights, deltas);
    }
    
    public double[] getErrorsParallel() {
        return NeuralMath.matmulParallel(weights, deltas);
    }
    
    /**
     * Aktualisiert die Gewichte durch Addieren der gespeicherten Gradienten.
     * Die Gradienten werden dabei zurückgesetzt.
     *
     * @param learningRate
     */
    public void updateWeights(double learningRate) {
        for(int conneuron = 0; conneuron < weights.length; conneuron++)
            for(int neuron = 0; neuron < weights[conneuron].length; neuron++) {
                weights[conneuron][neuron] += learningRate * gradients[conneuron][neuron];
                gradients[conneuron][neuron] = 0;
            }
        for(int b = 0; b < biases.length; b++) {
            biases[b] += learningRate * biasgradients[b];
            biasgradients[b] = 0;
        }
    }
    
    /**
     * Aktualisiert die Gewichte durch Addieren der gespeicherten Gradienten.
     * Die Gradienten werden dabei zurückgesetzt. Nutzt parallele Streams.
     *
     * @param learningRate
     */
    public void updateWeightsParallel(double learningRate) {
        IntStream.range(0, weights.length)
                .parallel()
                .forEach(conneuron -> {
                        for(int neuron = 0; neuron < weights[conneuron].length; neuron++) {
                            weights[conneuron][neuron] += learningRate * gradients[conneuron][neuron];
                            gradients[conneuron][neuron] = 0;
                        }
                });
        for(int b = 0; b < biases.length; b++) {
            biases[b] += learningRate * biasgradients[b];
            biasgradients[b] = 0;
        }
    }
    
}
