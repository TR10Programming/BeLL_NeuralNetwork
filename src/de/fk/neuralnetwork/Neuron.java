package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import java.io.Serializable;

/**
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
     * Gibt die Aktivierung des Neurons zurück.
     * 
     * Die Aktivierung wird nicht neu berechnet, dafür muss die trigger()-Methode
     * aufgerufen werden.
     *
     * @return Aktivierung des Neurons
     * @see Neuron#trigger(double[], de.fk.neuralnetwork.math.ActivationFunction) 
     */
    public double getActivation();
    
    /**
     * Weist dem Neuron einen Wert zu, der den Fehleranteil des Neurons an der
     * Ausgabe angibt.
     * 
     * Kommt bei der Backpropagation zum Einsatz. Sollte nur für das Output
     * Layer verwendet werden, andernfalls sollten die entsprechenden
     * Methoden zur Berechnung des Fehlers aus nachfolgenden Layern verwendet
     * werden.
     *
     * @param error Fehlerrate &delta; des Neurons
     */
    public void setError(double error);

    /**
     * Gibt die Fehlerrate des Neurons zurück.
     * 
     * Kommt bei der Backpropagation zum Einsatz.
     *
     * @return Fehlerrate &delta; des Neurons
     */
    public double getError();
    
    /**
     * Gibt den Delta-Wert (Produkt aus Fehler und Wert der ersten Ableitung der
     * Aktivierungsfunktion der Aktivierung) zurück.
     * 
     * Kommt bei der Backpropagation zum Einsatz.
     *
     * @param act Aktivierungsfunktion
     * @return
     */
    public double getErrorDelta(ActivationFunction act);
    
    /**
     * Gibt die Aktivierung des Neurons ohne Eingabe in die Aktivierungsfunktion
     * zurück. Es handelt sich also um die Summe aller Eingabesignale.
     * 
     * Die Summe wird nicht neu berechnet, dafür muss die trigger()-Methode
     * aufgerufen werden.
     *
     * @return Gewichtete Eingabesumme
     */
    public double getWeightedInput();
    
    /**
     * Berechnet den Fehler des Neurons durch Backpropagation aus den Fehlern
     * der Neuronen des folgenden Layers.
     *
     * @param neuronPos Position des Neurons im aktuellen Layer
     * @param nextLayer Nächstes Layer
     */
    public void calcError(int neuronPos, NeuralLayer nextLayer);
    
}
