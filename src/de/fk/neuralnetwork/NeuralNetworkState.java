package de.fk.neuralnetwork;

import java.util.ArrayList;

/**
 *
 * @author Felix
 */
public class NeuralNetworkState {

    private final ArrayList<double[]> activations = new ArrayList<>();
    
    public void addLayerActivations(double[] act) {
        activations.add(act);
    }
    
    public double[] getLayerActivations(int layer) {
        return activations.get(layer);
    }
    
    public int getLayerCount() {
        return activations.size();
    }
    
    public double[] getOutput() {
        return activations.get(activations.size() - 1);
    }
    
}
