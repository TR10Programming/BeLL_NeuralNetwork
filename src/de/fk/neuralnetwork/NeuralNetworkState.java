package de.fk.neuralnetwork;

import java.util.ArrayList;

/**
 * Beschreibt die Aktivierungen aller Neuronen eines neuronalen Netzes.
 *
 * @author Felix
 */
public class NeuralNetworkState {

    private final ArrayList<double[]> activations = new ArrayList<>();
    
    /**
     * Fügt eine weitere Schicht mit den übergebenen Aktivierungen hinzu.
     * Sollte für jedes Layer genau einmal aufgerufen werden.
     *
     * @param act
     */
    public void addLayerActivations(double[] act) {
        activations.add(act);
    }
    
    /**
     * Gibt die Aktivierungen des Layers mit dem übergebenen Index zurück.
     *
     * @param layer
     * @return
     */
    public double[] getLayerActivations(int layer) {
        return activations.get(layer);
    }
    
    /**
     * Gibt die Anzahl der gespeicherten Layer zurück.
     *
     * @return
     */
    public int getLayerCount() {
        return activations.size();
    }
    
    /**
     * Gibt die Aktivierungen der Ausgabeneuronen zurück.
     *
     * @return
     */
    public double[] getOutput() {
        return activations.get(activations.size() - 1);
    }
    
}
