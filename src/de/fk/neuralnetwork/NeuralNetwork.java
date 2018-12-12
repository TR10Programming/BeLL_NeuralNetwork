package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.Serializable;

/**
 * Repräsentiert ein neuronales Netz.
 *
 * @author Felix
 */
public class NeuralNetwork implements Serializable {
    
    private static final long serialVersionUID = -336732427642969125L/*655591235934461712L*/;
    
    private NeuralLayer[] layers;
    private int inputNeurons;
    private boolean inputBias;
    
    public void sout() {
        System.out.println(serialVersionUID);
    }
    
    /**
     * Generiert ein neuronales Netz mit den übergebenen Anzahlen an Neuronen.
     * Aller Layer außer dem Output Layer erhalten zusätzlich einen Bias.
     * 
     * @param neurons Liste mit Anzahlen der Neuronen inkl. Ein- und Ausgabeschicht (mind. 1)
     */
    public NeuralNetwork(int... neurons) {
        layers = new NeuralLayer[neurons.length - 1];
        inputBias = true;
        //First Hidden Layer
        layers[0] = new NeuralLayer(neurons[0] + 1, neurons[1], true);
        //Other Hidden Layers
        for(int i = 1; i < neurons.length - 2; i++) layers[i] = new NeuralLayer(neurons[i] + 1, neurons[i + 1], true);
        //Output Layer
        layers[neurons.length - 2] = new NeuralLayer(neurons[neurons.length - 2] + 1, neurons[neurons.length - 1], false);
        layers[neurons.length - 2].setActivationFunction(ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        this.inputNeurons = neurons[0];
    }
    
    /**
     * Generiert ein neuronales Netz mit der übergebenen Anzahl an
     * Eingabeneuronen und den übergebenen Layern, optional mit Input Bias.
     *
     * @param inputNeurons Anzahl Eingabeneuronen
     * @param inputBias Ob dem Input Layer ein Biasneuron hinzugefügt werden soll
     * @param layers Layers
     */
    public NeuralNetwork(int inputNeurons, boolean inputBias, NeuralLayer... layers) {
        this.inputNeurons = inputNeurons;
        this.inputBias = inputBias;
        this.layers = layers;
    }
    
    /**
     * Gibt das Output-Layer (das letzte Layer) zurück.
     *
     * @return Output Layer
     */
    public NeuralLayer getOutputLayer() {
        return layers[layers.length - 1];
    }
    
    public NeuralLayer[] getLayers() {
        return layers;
    }

    /**
     * Gibt die Anzahl der Eingabeneuronen zurück.
     *
     * @return Anzahl Eingabeneuronen
     */
    public int getInputNeurons() {
        return inputNeurons;
    }

    public boolean isInputBias() {
        return inputBias;
    }
    
    public int[] getNeuronCounts() {
        int[] neuronCounts = new int[layers.length + 1];
        neuronCounts[0] = inputNeurons + (inputBias ? 1 : 0);
        for(int i = 0; i < layers.length; i++) neuronCounts[i + 1] = layers[i].getNeurons().length;
        return neuronCounts;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen und gibt alle
     * Aktivierungen sowie die Ausgabe zurück.
     *
     * @param in Eingabeaktivierungen
     * @return Aktivierungen
     * @see NeuralNetworkState
     */
    public NeuralNetworkState trigger(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] vals = NeuralMath.addBias(in);
        NeuralNetworkState state = new NeuralNetworkState();
        for(NeuralLayer layer : layers) state.addLayerActivations(vals = layer.trigger(vals));
        return state;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen und gibt alle
     * Aktivierungen sowie die Ausgabe zurück. (Nutzt parallele Streams)
     *
     * @param in Eingabeaktivierungen
     * @return Aktivierungen
     * @see NeuralNetworkState
     */
    public NeuralNetworkState triggerParallel(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] vals = NeuralMath.addBias(in);
        NeuralNetworkState state = new NeuralNetworkState();
        for(NeuralLayer layer : layers) state.addLayerActivations(vals = layer.triggerParallel(vals));
        return state;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen und gibt nur die
     * Ausgabe zurück. (Nutzt parallele Streams)
     *
     * @param in Eingabeaktivierungen
     * @return Ausgabeaktivierungen
     */
    public double[] getOutputParallel(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] vals = NeuralMath.addBias(in);
        for(NeuralLayer layer : layers) vals = layer.triggerParallel(vals);
        return vals;
    }
    
    /**
     * Muss vor jedem Start des Backpropagation-Algorithmus aufgerufen werden.
     * Setzt alle gespeicherten Gewichtsänderungen zurück und initialisiert sie
     * neu für die übergebene Anzahl an Threads.
     *
     * @param threads
     */
    public void prepareParallelBackprop(int threads) {
        for(NeuralLayer layer : layers) layer.prepareForParallelBackprop(threads);
    }

}
