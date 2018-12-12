package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.Serializable;

/**
 * Repräsentiert ein Neuron in einem neuronalen Netz. Ein Neuron ist die
 * kleinste Einheit im neuronalen Netz. Es kann durch ein Eingabesignal
 * aktiviert werden (siehe trigger()) und seinen Fehler (siehe getError()) bzw.
 * Delta-Fehler (siehe getErrorDelta()) für den Backpropagation-Algorithmus
 * ermitteln. Ein klassisches Neuron wird durch die BasicNeuron-Klasse
 * respräsentiert und besitzt Gewichte zu allen Vorgängerneuronen, das
 * Bias-Neuron hat hingegen keine klassischen Neuroneneigenschaften, sondern
 * dient als Platzhalter für einen Bias-Parameter.
 *
 * @author Felix
 * @see Neuron#trigger(double[], de.fk.neuralnetwork.math.ActivationFunction) trigger(..)
 * @see Neuron#getError(int, de.fk.neuralnetwork.NeuralLayer, double[]) getError(..)
 * @see Neuron#getErrorDelta(double, de.fk.neuralnetwork.math.ActivationFunction, double[]) getErrorDelta(..)
 * @see NeuralLayer NeuralLayer
 * @see BasicNeuron
 * @see BiasNeuron
 */
public interface Neuron extends Serializable {
    
    /**
     * Gibt den Ausgabewert (Activation) des Neurons zurück.
     * 
     * Bei Aufruf der Funktion wird die Aktivierung des Neurons geupdatet und
     * kann mit getActivation() abgerufen werden.
     *
     * @param input Eingabevektor
     * @param act Zu verwendende Aktivierungsfunktion
     * @return Ausgabewert
     * @see Neuron#getActivation() 
     * @see NeuralMath#applyWeights(double[], double[]) 
     */
    public double trigger(double[] input, ActivationFunction act);
    
    /**
     * Gibt den Delta-Wert (Produkt aus Fehler und Wert der ersten Ableitung der
     * Aktivierungsfunktion der Aktivierung) zurück.
     * 
     * Kommt bei der Backpropagation zum Einsatz.
     *
     * @param error Fehler
     * @param act Aktivierungsfunktion
     * @param activationsBefore Aktivierungen im vorherigen Layer
     * @return
     * @see NeuralMath#applyWeights(double[], double[]) 
     */
    public double getErrorDelta(double error, ActivationFunction act, double[] activationsBefore);
    
    /**
     * Berechnet den Fehler des Neurons durch Backpropagation aus den Fehlern
     * der Neuronen des folgenden Layers.
     *
     * @param neuronPos Position des Neurons im aktuellen Layer
     * @param nextLayer Nächstes Layer
     * @param errorDeltasNextLayer Error Delta Werte des nächsten Layers
     * @return Fehler
     */
    public double getError(int neuronPos, NeuralLayer nextLayer, double[] errorDeltasNextLayer);
    
}
