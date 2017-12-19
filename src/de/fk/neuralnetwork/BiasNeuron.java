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
    public double getActivation() {
        return 1;
    }

    @Override
    public void setError(double error) {
        throw new UnsupportedOperationException("Ein Bias kann keinen Fehleranteil delta besitzen, da er keine ihm vorausgehenden Verbindungen hat.");
    }

    @Override
    public double getError() {
        throw new UnsupportedOperationException("Ein Bias kann keinen Fehleranteil delta besitzen, da er keine ihm vorausgehenden Verbindungen hat.");
    }

    @Override
    public double getErrorDelta(ActivationFunction act) {
        throw new UnsupportedOperationException("Ein Bias kann keinen Delta-Wert besitzen, da er keine ihm vorausgehenden Verbindungen hat."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double getWeightedInput() {
        throw new UnsupportedOperationException("Ein Bias hat keine Eingabesignale.");
    }

    @Override
    public void calcError(int neuronPos, NeuralLayer nextLayer) {
        //Es ist unnötig, den Fehler eines Bias-Neurons zu berechnen, da es keine Eingabesignale besitzt.
    }
    
}
