package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.Serializable;

/**
 * Repräsentiert ein Neuron in einem neuronalen Netz.
 *
 * @author Felix
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
