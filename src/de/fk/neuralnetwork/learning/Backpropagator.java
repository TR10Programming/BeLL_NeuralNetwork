package de.fk.neuralnetwork.learning;

import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.training.TrainingSupplier;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.math.NeuralMath;
import de.fk.neuralnetwork.training.TrainingExample;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class Backpropagator {

    private NeuralNetwork net;
    private double learningRate, regularizationRate;
    
    public Backpropagator(NeuralNetwork net, double learningRate, double regularizationRate) {
        this.net = net;
        this.learningRate = learningRate;
        this.regularizationRate = regularizationRate;
    }
    
    public void train(TrainingSupplier trainingSupplier, int iterations, OutputStream errorLoggerStream, int logEveryXIterations) {
        int exampleCount = trainingSupplier.getExampleCount(), logIt = logEveryXIterations;
        double error = 0.0;
        System.out.println("Training with " + exampleCount + " examples per iteration.");
        for(int iteration = 0; iteration < iterations; iteration++, logIt++) {
            error = 0.0;
            for(int example = 0; example < exampleCount; example++) {
                error += backpropStep(trainingSupplier.nextTrainingExample());
            }
            error /= exampleCount;
            if(errorLoggerStream != null && logIt == logEveryXIterations) try {
                logIt = 0;
                errorLoggerStream.write((iteration + " " + error + "\r\n").getBytes("UTF-8"));
            } catch (IOException ex) {}
        }
        System.out.println("Trained for " + iterations + " iterations. Error: " + error);
    }
    
    /**
     * Führt einen Backprop-Schritt aus und gibt den Fehler zurück.
     *
     * @param trainingExample
     * @return
     */
    public double backpropStep(TrainingExample trainingExample) {
        NeuralLayer[] layers = net.getLayers();
        //Sammeln der Trainingsdaten
        double[] input = trainingExample.getIn(), expectedOutput = trainingExample.getOut(),
        //Aktivierungen berechnen
        output = net.trigger(input);
        //System.out.println("Training for " + Arrays.toString(input) + " -> " + Arrays.toString(expectedOutput));
        double[] errorDeltas;
        //Output Layer
        NeuralLayer outputLayer = layers[layers.length - 1];
        outputLayer.setErrors(NeuralMath.getErrors(output, expectedOutput));
        outputLayer.calcAccumulatorMatrices(layers.length > 1 ? layers[layers.length - 2].getActivations() : input);
        //Hidden Layer
        for(int i = layers.length - 2; i >= 0; i--) {
            layers[i].calcErrors(outputLayer);
            layers[i].calcAccumulatorMatrices(i > 0 ? layers[i - 1].getActivations() : NeuralMath.addBias(input));
        }
        //Learn/Apply accumulators
        Arrays.stream(layers).parallel().forEach(l -> l.accumulate(learningRate, regularizationRate));
        //...
        //errorDeltas = outputLayer.getErrorDeltas();
        //Fehler berechnen
        return NeuralMath.getRegularizedError(output, expectedOutput, regularizationRate, net);
    }
    
}
