package de.fk.neuralnetwork;

import de.fk.neuralnetwork.math.ActivationFunction;
import de.fk.neuralnetwork.math.NeuralMath;

/**
 * Ein Neuron mit Gewichten zu seinem Vorgänger.
 *
 * @author Felix
 */
public class BasicNeuron implements Neuron {
    
    private static final long serialVersionUID = 4609093747635559427L/*8124483782436155978L*/;

    private double[] weights, weightsChange;
    private double[][] accum;
    
    /**
     * Generiert ein Neuron mit der übergebenen Anzahl an verbundenen Neuronen
     * im vorhergehenden Layer.
     *
     * @param connectedNeurons
     */
    public BasicNeuron(int connectedNeurons) {
        weights = NeuralMath.generateRandomWeights(connectedNeurons);
        this.accum = new double[1][weights.length];
        this.weightsChange = new double[weights.length];
        //System.out.println("Neues Neuron mit " + connectedNeurons + " Vorgängern generiert.");
    }
    
    /**
     * Generiert ein Neuron mit den übergebenen Gewichten.
     *
     * @param weights Gewichte
     */
    public BasicNeuron(double[] weights) {
        this.weights = weights;
        this.accum = new double[1][weights.length];
        this.weightsChange = new double[weights.length];
    }
    
    /**
     * Berechnet den Ausgabewert des Neurons anhand der Gewichte des Neurons,
     * dem gegebenen Eingabevektor und der Aktivierungsfunktion.
     *
     * @param input Eingabevektor
     * @param act Zu verwendende Aktivierungsfunktion
     * @return Ausgabewert
     * @see NeuralMath#applyWeights(double[], double[]) 
     */
    @Override
    public double trigger(double[] input, ActivationFunction act) {
        return act.apply(NeuralMath.applyWeights(input, weights));
    }
    
    public double[] getWeights() {
        return weights;
    }
    
    public double getWeightAt(int pos) {
        return weights[pos];
    }

    @Override
    public double getErrorDelta(double error, ActivationFunction act, double[] activationsBefore) {
        return error * act.derivative(NeuralMath.applyWeights(activationsBefore, weights));
    }
    
    /**
     * Setzt die gespeicherten Gewichtsänderungen zurück. Sollte normalerweise
     * nicht manuell aufgerufen werden.
     *
     * @param threadId ID des aktiven Threads
     */
    public void resetAccumulatorMatrix(int threadId) {
        this.accum[threadId] = new double[weights.length];
    }
    
    /**
     * Berechnet die Gewichtsänderungen für die übergebenen Aktivierungen und
     * Delta-Fehler und speichert diese.
     *
     * @param errorDelta Delta-Fehler
     * @param activationsBefore Aktivierungen im vorhergehenden Layer
     * @param act Aktivierungsfunktion
     * @param threadId Thread-ID
     * @see BasicNeuron#accumulate(double, double, double) 
     */
    public void calcAccumulatorMatrix(double errorDelta, double[] activationsBefore, ActivationFunction act, int threadId) {
        for(int i = 0; i < accum[threadId].length; i++) accum[threadId][i] += errorDelta * activationsBefore[i];
    }
    
    /**
     * Updatet die Gewichte, nachdem <code>calcAccumulatorMatrix</code>
     * mindestens einmal aufgerufen wurde. Anschließend werden die gespeicherten
     * Gewichtsänderungen zurückgesetzt.
     *
     * @param learningRate Lernrate Alpha
     * @param regularizationRate Regularisierungsrate Lambda
     * @param momentum Trägheit My
     * @see BasicNeuron#calcAccumulatorMatrix(double, double[], de.fk.neuralnetwork.math.ActivationFunction, int) 
     */
    public void accumulate(double learningRate, double regularizationRate, double momentum) {//TODO Regularization
        for(int i = 0; i < weights.length; i++) {
            double weightsChangeBefore = weightsChange[i];
            //Werte aus allen Threads aufsummieren
            weightsChange[i] = 0.0;
            for(double[] acct : accum) {
                weightsChange[i] += acct[i];
            }
            if(weightsChangeBefore == 0.0) weights[i] += learningRate * weightsChange[i];
            else weights[i] += (1 - momentum) * (learningRate * weightsChange[i]) + momentum * weightsChangeBefore;
        }
        //Reset
        this.accum = new double[accum.length][weights.length];
    }

    @Override
    public double getError(int neuronPos, NeuralLayer nextLayer, double[] errorDeltasNextLayer) {
        double error = 0;
        Neuron[] neuronsNextLayer = nextLayer.getNeurons();
        for(int i = 0, j = 0; i < errorDeltasNextLayer.length && j < neuronsNextLayer.length; i++, j++) {
            while(!(neuronsNextLayer[j] instanceof BasicNeuron)) j++;
            error += errorDeltasNextLayer[i] * ((BasicNeuron) neuronsNextLayer[j]).getWeightAt(neuronPos);
        }
        return error;
    }
    
    /**
     * Muss bei jedem Start des Backpropagation-Algorithmus aufgerufen werden.
     *
     * @param threads Anzahl der Threads
     */
    public void prepareForParallelBackprop(int threads) {
        this.accum = new double[threads][weights.length];
    }
    
}
