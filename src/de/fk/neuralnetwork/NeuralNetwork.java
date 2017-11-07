package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class NeuralNetwork {
    
    private NeuralLayer[] layers;
    private int inputNeurons, outputNeurons;
    
    public NeuralNetwork(int hiddenLayersCount, int neuronsPerLayer) {
        layers = new NeuralLayer[hiddenLayersCount];
        for(int i = 0; i < hiddenLayersCount; i++) layers[i] = new NeuralLayer(neuronsPerLayer, neuronsPerLayer);
        this.inputNeurons = neuronsPerLayer;
        this.outputNeurons = neuronsPerLayer;
    }
    
    public NeuralNetwork(int hiddenLayersCount, int neuronsPerHiddenLayer, int inputNeurons, int outputNeurons) {
        layers = new NeuralLayer[hiddenLayersCount + 1];
        //Input Layer
        layers[0] = new NeuralLayer(inputNeurons, neuronsPerHiddenLayer);
        //Hidden Layers
        for(int i = 1; i < hiddenLayersCount; i++) layers[i] = new NeuralLayer(neuronsPerHiddenLayer, neuronsPerHiddenLayer);
        //Output Layer
        layers[hiddenLayersCount] = new NeuralLayer(neuronsPerHiddenLayer, outputNeurons);
        layers[hiddenLayersCount].setActivationFunction(ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        this.inputNeurons = inputNeurons;
        this.outputNeurons = outputNeurons;
    }
    
    public NeuralLayer getInputLayer() {
        return layers[0];
    }
    
    public NeuralLayer getOutputLayer() {
        return layers[layers.length - 1];
    }
    
    public double[] trigger(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        System.out.println("Eingabe: " + Arrays.toString(in));
        double[] vals = in.clone();
        for(int i = 0; i < layers.length; i++) {
            vals = layers[i].trigger(vals);
            System.out.println("Layer " + i + ": " + Arrays.toString(vals));
        }
        System.out.println("Ausgabe: " + Arrays.toString(vals));
        return vals;
    }

}
