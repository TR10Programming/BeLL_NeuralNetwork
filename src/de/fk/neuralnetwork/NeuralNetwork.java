package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.Serializable;

/**
 *
 * @author Felix
 */
public class NeuralNetwork implements Serializable {
    
    private static final long serialVersionUID = /*-336732427642969125L*/655591235934461712L;
    
    private NeuralLayer[] layers;
    private int inputNeurons, outputNeurons;
    private boolean inputBias;
    
    public void sout() {
        System.out.println(serialVersionUID);
    }
    
    /**
     * Generiert ein neuronales Netz mit der gegebenen Anzahl an Layern und
     * Neuronen pro Layer (für alle Layer gleich). Alle Layer außer dem Output
     * Layer erhalten zusätzlich einen Bias.
     *
     * @param layerCount
     * @param neuronsPerLayer
     */
    public NeuralNetwork(int layerCount, int neuronsPerLayer) {
        layers = new NeuralLayer[layerCount];
        inputBias = true;
        //First Hidden Layer
        layers[0] = new NeuralLayer(inputNeurons + 1, neuronsPerLayer, true);
        //Other Hidden Layers
        for(int i = 1; i < layerCount - 1; i++) layers[i] = new NeuralLayer(neuronsPerLayer + 1, neuronsPerLayer, true);
        //Output Layer
        layers[layerCount - 1] = new NeuralLayer(neuronsPerLayer + 1, neuronsPerLayer, false);
        layers[layerCount - 1].setActivationFunction(ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        this.inputNeurons = neuronsPerLayer;
        this.outputNeurons = neuronsPerLayer;
    }
    
    /**
     * Generiert ein neuronales Netz mit einem Input Layer, der übergebenen Anzahl
     * an Hidden Layer und einem Output Layer mit der übergebenen Anzahl an Neuronen
     * pro Hidden, Input und Output Layer. Alle Layer außer dem Output Layer
     * erhalten zusätzlich einen Bias.
     *
     * @param hiddenLayersCount
     * @param neuronsPerHiddenLayer
     * @param inputNeurons
     * @param outputNeurons
     */
    public NeuralNetwork(int hiddenLayersCount, int neuronsPerHiddenLayer, int inputNeurons, int outputNeurons) {
        layers = new NeuralLayer[hiddenLayersCount + 1];
        inputBias = true;
        //First Hidden Layer
        layers[0] = new NeuralLayer(inputNeurons + 1, neuronsPerHiddenLayer, true);
        //Other Hidden Layers
        for(int i = 1; i < hiddenLayersCount; i++) layers[i] = new NeuralLayer(neuronsPerHiddenLayer + 1, neuronsPerHiddenLayer, true);
        //Output Layer
        layers[hiddenLayersCount] = new NeuralLayer(neuronsPerHiddenLayer + 1, outputNeurons, false);
        layers[hiddenLayersCount].setActivationFunction(ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        this.inputNeurons = inputNeurons;
        this.outputNeurons = outputNeurons;
    }
    
    /**
     * Generiert ein neuronales Netz mit den übergebenen Anzahlen an Neuronen.
     * Aller Layer außer dem Output Layer erhalten zusätzlich einen Bias.
     *
     * @param inputNeurons Anzahl der Eingabeneuronen
     * @param outputNeurons Anzahl der Ausgabeneuronen
     * @param neurons Liste mit Anzahlen der Hidden Layer Neuronen (mind. 1)
     */
    public NeuralNetwork(int inputNeurons, int outputNeurons, int... neurons) {
        layers = new NeuralLayer[neurons.length + 1];
        inputBias = true;
        //First Hidden Layer
        layers[0] = new NeuralLayer(inputNeurons + 1, neurons[0], true);
        //Other Hidden Layers
        for(int i = 1; i < neurons.length; i++) layers[i] = new NeuralLayer(neurons[i - 1] + 1, neurons[i], true);
        //Output Layer
        layers[neurons.length] = new NeuralLayer(neurons[neurons.length - 1] + 1, outputNeurons, false);
        layers[neurons.length].setActivationFunction(ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        this.inputNeurons = inputNeurons;
        this.outputNeurons = outputNeurons;
    }
    
    public NeuralLayer getOutputLayer() {
        return layers[layers.length - 1];
    }
    
    public NeuralLayer[] getLayers() {
        return layers;
    }
    
    public double[] trigger(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] vals = NeuralMath.addBias(in);
        //System.out.println("Eingabe (Layer 1): " + Arrays.toString(vals));
        for(int i = 0; i < layers.length; i++) {
            vals = layers[i].trigger(vals);
            //System.out.println("Layer " + (i + 2) + ": " + Arrays.toString(vals));
        }
        //System.out.println("Ausgabe: " + Arrays.toString(vals));
        return vals;
    }

}
