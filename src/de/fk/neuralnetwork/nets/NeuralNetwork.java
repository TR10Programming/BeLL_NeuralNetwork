package de.fk.neuralnetwork.nets;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.LossFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import de.fk.neuralnetwork.training.TrainingExample;
import javafx.util.Pair;

/**
 * Repräsentiert ein neuronales Netz.
 *
 * @author Felix
 */
@Deprecated
public class NeuralNetwork {
    
    private NeuralLayer[] layers;
    private int inputNeurons;
    private LossFunction lossFunction = new LossFunction.MSE();
    
    /**
     * Generiert ein neuronales Netz mit den übergebenen Anzahlen an Neuronen.
     *
     * @param inputNeurons
     * @param neurons Liste mit Anzahlen der Hidden Layer Neuronen (mind. 1)
     */
    public NeuralNetwork(int inputNeurons, int... neurons) {
        if(neurons.length < 1) throw new IllegalArgumentException("Wenigstens eine Schicht benötigt.");
        this.inputNeurons = inputNeurons;
        layers = new NeuralLayer[neurons.length];
        //First Hidden Layer
        layers[0] = new NeuralLayer(inputNeurons, neurons[0], neurons.length > 1 ? ActivationFunction.DEFAULT_ACTIVATION_FUNCTION : ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
        //Other Hidden Layers
        for(int i = 1; i < neurons.length - 1; i++) layers[i] = new NeuralLayer(neurons[i - 1], neurons[i], ActivationFunction.DEFAULT_ACTIVATION_FUNCTION);
        //Output Layer
        if(neurons.length > 1) layers[neurons.length - 1] = new NeuralLayer(neurons[neurons.length - 2], neurons[neurons.length - 1], ActivationFunction.DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION);
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
    
    public int[] getNeuronCounts() {
        int[] neuronCounts = new int[layers.length + 1];
        neuronCounts[0] = inputNeurons;
        for(int i = 0; i < layers.length; i++) neuronCounts[i + 1] = layers[i].getNeuronCount();
        return neuronCounts;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen. Die Aktivierungen
     * können über die einzelnen Schichten abgerufen werden.
     *
     * @param in Eingabeaktivierungen
     * @return Aktivierungen der Ausgabeschicht
     * @see NeuralNetworkState
     */
    public double[] trigger(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] activations = in;
        for(NeuralLayer layer : layers) activations = layer.trigger(activations);
        return activations;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen. Die Aktivierungen
     * können über die einzelnen Schichten abgerufen werden. Nutzt parallele
     * Streams.
     *
     * @param in Eingabeaktivierungen
     * @return Aktivierungen der Ausgabeschicht
     * @see NeuralNetworkState
     */
    public double[] triggerParallel(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] activations = in;
        for(NeuralLayer layer : layers) activations = layer.triggerParallel(activations);
        return activations;
    }
    
    /**
     * Lässt die Eingabedaten das neuronale Netz durchlaufen, ohne Aktivierungen
     * zwischenzuspeichern. Nutzt parallele Streams. Schneller als
     * triggerParallel.
     *
     * @param in
     * @return
     */
    public double[] getOutput(double[] in) {
        if(in.length != inputNeurons) throw new IllegalArgumentException("Es gibt " + inputNeurons + " Eingabeneuronen, es wurden aber " + in.length + " Werte eingegeben.");
        double[] activations = in;
        for(NeuralLayer layer : layers) activations = layer.getOutput(activations);
        return activations;
    }
    
    /**
     * Führt einen Trainingszyklus im Stochastic Gradient Descent aus.
     *
     * @param ex
     * @param learningRate
     */
    public void stochasticBackprop(TrainingExample ex, double learningRate) {
        triggerParallel(ex.getIn());
        int l = layers.length - 1;
        //Backprop Output Layer
        layers[l].backpropParallel(lossFunction.derivative(layers[l].getActivations(), ex.getOut()), l > 0 ? layers[l - 1].getActivations() : ex.getIn());
        //Backprop Hidden Layers
        for(l--; l >= 0; l--)
            layers[l].backpropParallel(layers[l + 1].getErrorsParallel(), l > 0 ? layers[l - 1].getActivations() : ex.getIn());
        //Gewichte updaten
        for(l = 0; l < layers.length; l++) layers[l].updateWeightsParallel(learningRate);
    }
    
    public Pair<Double, Boolean> getLossAndAccuracy(double[] in, double[] expectedOut) {
       double[] out = getOutput(in);
        return new Pair<>(lossFunction.apply(out, expectedOut),
                NeuralMath.getPredictedLabel(out) == NeuralMath.getPredictedLabel(expectedOut));
    }

}
