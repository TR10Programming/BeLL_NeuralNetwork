package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * Eine Schicht mit einer beliebigen Anzahl Neuronen.
 *
 * @author Felix
 */
public class NeuralLayer implements Serializable {
    
    private static final long serialVersionUID = -7304433412748964606L/*-7456758553687520539L*/;
    
    private Neuron[] neurons;
    private ActivationFunction act;
    
    /**
     * Erstellt eine neue Neuronenschicht.
     *
     * @param connectedNeurons Anzahl der Vorgängerneuronen
     * @param neuronCount Anzahl der Neuronen
     * @param bias true, die Schicht ein Bias-Neuron enthalten soll
     */
    public NeuralLayer(int connectedNeurons, int neuronCount, boolean bias) {
        //System.out.println("Neues Layer generiert mit " + connectedNeurons + " Vorgängern und " + neuronCount + " Neuronen");
        neurons = new Neuron[neuronCount + (bias ? 1 : 0)];
        if(bias) neurons[0] = new BiasNeuron();
        for(int i = (bias ? 1 : 0); i < neurons.length; i++) neurons[i] = new BasicNeuron(connectedNeurons);
        act = ActivationFunction.DEFAULT_ACTIVATION_FUNCTION;
    }
    
    /**
     * Erstellt eine neue Neuronenschicht aus vorhandenen Neuronen.
     *
     * @param act Aktivierungsfunktion aller Neuronen dieser Schicht
     * @param neurons Array mit Neuronen dieser Schicht
     */
    public NeuralLayer(ActivationFunction act, Neuron... neurons) {
        this.act = act;
        this.neurons = neurons;
    }

    /**
     * Gibt die in dieser Neuronenschicht enthaltenen Neuronen zurück.
     *
     * @return
     */
    public Neuron[] getNeurons() {
        return neurons;
    }

    /**
     * Ändert die Aktivierungsfunktion der gesamten Schicht.
     *
     * @param act
     */
    public void setActivationFunction(ActivationFunction act) {
        this.act = act;
    }

    /**
     * Gibt die von allen Neuronen dieser Schicht verwendete
     * Aktivierungsfunktion zurück.
     *
     * @return
     */
    public ActivationFunction getActivationFunction() {
        return act;
    }
    
    /**
     * Aktiviert die Neuronen dieser Schicht mit den übergebenen Eingabesignalen
     * und gibt die Ausgabesignale zurück.
     *
     * @param in Eingabesignale
     * @return Ausgabesignale
     */
    public double[] trigger(double[] in) {
        return Arrays.stream(neurons).mapToDouble(n -> n.trigger(in, act)).toArray();
    }
    
    /**
     * Aktiviert die Neuronen dieser Schicht mit den übergebenen Eingabesignalen
     * und gibt die Ausgabesignale zurück. Verwendet parallele Streams.
     *
     * @param in Eingabesignale
     * @return Ausgabesignale
     */
    public double[] triggerParallel(double[] in) {
        return Arrays.stream(neurons).parallel().mapToDouble(n -> n.trigger(in, act)).toArray();
    }
    
    /**
     * Berechnet die Delta-Fehler aller Neuronen dieser Schicht und gibt diese
     * als Array zurück.
     *
     * @see Neuron#getErrorDelta(double, de.fk.neuralnetwork.math.ActivationFunction, double[]) Neuron.getErrorDelta(...)
     * @param errors Fehlerarray der Neuronen dieser Schicht
     * @param activationsBefore Array mit Aktivierungen der Neuronen aus der vorhergehenden Schicht
     * @return
     */
    public double[] getErrorDeltas(double[] errors, double[] activationsBefore) {
        ArrayList<Double> errorDeltas = new ArrayList<>();
        for(int i = 0, j = 0; i < errors.length && j < neurons.length; i++, j++) {
            while(!(neurons[j] instanceof BasicNeuron) && j < neurons.length) j++;
            if(j >= neurons.length) break;
            errorDeltas.add(neurons[j].getErrorDelta(errors[i], act, activationsBefore));
        }
        return errorDeltas.stream().mapToDouble(Double::doubleValue).toArray();
    }
    
    /**
     * Speichert die Änderungen der Hyperparameter und verrechnet diese, ohne
     * diese dabei upzudaten. Die Änderungen der Hyperparameter werden in den
     * sog. Accumulator Matrices gespeichert. Je nach verwendetem
     * Trainingsalgorithmus werden die Hyperparameter nach jeder Iteration
     * (Stochastic Gradient Descent) oder erst nach n Iterationen (Batch
     * Gradient Descent) geupdatet. 
     *
     * @param errorDeltas Delta-Fehler dieser Schicht
     * @param activationsBefore Aktivierungen der vorhergehenden Schicht
     * @param threadId Bei Verwendung von paralleler Backpropagation: Die Nummer des Threads, der die Änderungen vornimmt
     * @see NeuralLayer#accumulate(double, double, double) accumulate(..)
     */
    public void calcAccumulatorMatrices(double[] errorDeltas, double[] activationsBefore, int threadId) {
        for(int i = 0, j = 0; i < errorDeltas.length && j < neurons.length; i++, j++) {
            while(!(neurons[j] instanceof BasicNeuron) && j < neurons.length) j++;
            if(j >= neurons.length) break;
            ((BasicNeuron) neurons[j]).calcAccumulatorMatrix(errorDeltas[i], activationsBefore, act, threadId);
        }
        //IntStream.range(0, neurons.length).filter(i -> neurons[i] instanceof BasicNeuron).forEach(i -> ((BasicNeuron) neurons[i]).calcAccumulatorMatrix(errors[i], activationsBefore, act, threadId));
    }
    
    /**
     * Berechnet die Fehler jedes Neurons ausgehend von den Delta-Fehlern der
     * Neuronen der darauffolgenden Schicht und gibt diese als Array zurück.
     *
     * @param nextLayer Nächste Neuronenschicht
     * @param errorDeltasNextLayer Delta-Fehlerarray der nächsten Schicht
     * @return
     */
    public double[] getErrors(NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        return IntStream.range(0, neurons.length).filter(i -> neurons[i] instanceof BasicNeuron).mapToDouble(i -> neurons[i].getError(i, nextLayer, errorDeltasNextLayer)).toArray();
    }
    
    /**
     * Berechnet die Fehler jedes Neurons ausgehend von den Delta-Fehlern der
     * Neuronen der darauffolgenden Schicht und gibt diese als Array zurück.
     * Verwendet parallele Streams.
     *
     * @param nextLayer Nächste Neuronenschicht
     * @param errorDeltasNextLayer Delta-Fehlerarray der nächsten Schicht
     * @return
     */
    public double[] getErrorsParallel(NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        return IntStream.range(0, neurons.length).parallel().filter(i -> neurons[i] instanceof BasicNeuron).mapToDouble(i -> neurons[i].getError(i, nextLayer, errorDeltasNextLayer)).toArray();
    }
    
    /**
     * Updatet die Hyperparameter mit den in den Accumulator Matrices
     * gespeicherten Werten. Die Änderungen der Hyperparameter werden in den
     * sog. Accumulator Matrices gespeichert. Je nach verwendetem
     * Trainingsalgorithmus werden die Hyperparameter nach jeder Iteration
     * (Stochastic Gradient Descent) oder erst nach n Iterationen (Batch
     * Gradient Descent) geupdatet. 
     *
     * @param learningRate
     * @param regularizationRate
     * @param momentum
     * @see NeuralLayer#calcAccumulatorMatrices(double[], double[], int) calcAccumulatorMatrices(..)
     */
    public void accumulate(double learningRate, double regularizationRate, double momentum) {
        Arrays.stream(neurons)
                .parallel()
                .filter(n -> n instanceof BasicNeuron)
                .forEach(n -> ((BasicNeuron) n).accumulate(learningRate, regularizationRate, momentum));
    }
    
    /**
     * Bereitet die Neuronen für parallele Backpropagation vor. Um mehrere
     * Backpropagation-Vorgänge gleichzeitig durchführen zu können, müssen alle
     * Threads getrennte Accumulator Matrices besitzen. Erst beim Updaten der
     * Gewichte werden die Werte aus allen Accumulator Matrices zusammengeführt.
     * Vor Beginn paralleler Backpropagation muss durch Aufruf dieser Methode
     * einmalig die Anzahl der bereitzustellenden Accumulator Matrices
     * festgelegt werden.
     *
     * @param threads Anzahl der verwendeten Threads
     * @see NeuralLayer#accumulate(double, double, double) accumulate(..)
     */
    public void prepareForParallelBackprop(int threads) {
        for(Neuron n : neurons) if(n instanceof BasicNeuron) ((BasicNeuron) n).prepareForParallelBackprop(threads);
    }

}
