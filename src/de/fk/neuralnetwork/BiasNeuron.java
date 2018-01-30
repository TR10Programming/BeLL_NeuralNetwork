package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;

/**
 *
 * @author Felix
 */
public class BiasNeuron implements Neuron {
    
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
