package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;

/**
 * Ein Neuron ohne Gewichte mit der Aktivierung 1. Für die Backpropagation
 * irrelevant. Dient als Platzhalter für einen Bias-Parameter und besitzt
 * als solcher keine Vorgänger und entsprechend keine Gewichte.
 *
 * @author Felix
 * @see Neuron
 */
public class BiasNeuron implements Neuron {
    
    /**
     * Erstellt ein neues Bias-Neuron.
     *
     */
    public BiasNeuron() {
        
    }
    
    @Override
    public double trigger(double[] input, ActivationFunction act) {
        return 1;
    }

    @Override
    public double getErrorDelta(double error, ActivationFunction act, double[] activationsBefore) {
        throw new UnsupportedOperationException("Ein Bias kann keinen Delta-Wert besitzen, da er keine ihm vorausgehenden Verbindungen hat."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getError(int neuronPos, NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        throw new UnsupportedOperationException("Es ist unmöglich, den Fehler eines Bias-Neurons zu berechnen, da es keine Eingabesignale besitzt.");
    }
    
}
