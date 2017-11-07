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
    
    public NeuralLayer(int connectedNeurons, int neuronCount) {
        System.out.println("Neues Layer generiert mit " + connectedNeurons + " Vorgängern und " + neuronCount + " Neuronen");
        neurons = new Neuron[neuronCount];
        for(int i = 0; i < neuronCount; i++) neurons[i] = new Neuron(connectedNeurons);
        act = ActivationFunction.DEFAULT_ACTIVATION_FUNCTION;
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

}
