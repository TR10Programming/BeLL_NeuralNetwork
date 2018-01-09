package de.fk.neuralnetwork.learning;

import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.training.TrainingSupplier;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.math.NeuralMath;
import de.fk.neuralnetwork.training.TrainingExample;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public class Backpropagator {

    private NeuralNetwork net;
    private double learningRate, regularizationRate;
    private Thread trainThread;
    private boolean trainThreadStopped;
    
    public Backpropagator(NeuralNetwork net, double learningRate, double regularizationRate) {
        this.net = net;
        this.learningRate = learningRate;
        this.regularizationRate = regularizationRate;
        this.trainThread = null;
        this.trainThreadStopped = false;
    }
    
    /**
     * Startet den Trainingsvorgang.
     *
     * @param trainingSupplier TrainingSupplier mit Trainingsbeispielen
     * @param iterations Wiederholungen aller Trainingsbeispiele
     * @param errorLoggerStream OutputStream auf dem Iteration und Fehlerrate durch Leerzeichen separiert zeilenweise ausgegeben werden
     * @param logEveryXIterations Bestimmt, aller wie vieler Iterationen eine Ausgabe erfolgen soll
     * @return Trainingsthread
     * @throws IllegalStateException Wenn bereits ein Trainingsthread läuft
     * @see Backpropagator#stopTraining() 
     */
    public Thread train(TrainingSupplier trainingSupplier, int iterations, OutputStream errorLoggerStream, int logEveryXIterations) throws IllegalStateException {
        if(trainThread != null) throw new IllegalStateException("Es wird bereits trainiert.");
        trainThreadStopped = false;
        (trainThread = new Thread(() -> {
        
            long startTime = System.currentTimeMillis();
            int exampleCount = trainingSupplier.getExampleCount(), logIt = logEveryXIterations;
            double error = 0.0;
            System.out.println("Training with " + exampleCount + " examples per iteration.");
            for(int iteration = 0; iteration < iterations; iteration++, logIt++) {
                if(trainThreadStopped) break;
                error = 0.0;
                for(int example = 0; example < exampleCount; example++) {
                    error += backpropStep(trainingSupplier.nextTrainingExample());
                }
                error /= exampleCount;
                if(logIt == logEveryXIterations) try {
                    logIt = 0;
                    double time = (System.currentTimeMillis() - startTime) / 1000.0;
                    byte[] out = (iteration + " " + error + " " + time + "\r\n").getBytes("UTF-8");
                    if(errorLoggerStream != null) errorLoggerStream.write(out);
                    System.out.write(out);
                    //Kopie speichern
                    ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File("300-100_" + iteration + ".net")));
                    oos.writeObject(net);
                    oos.close();
                } catch (IOException ex) {}
            }
            System.out.println("Trained for " + iterations + " iterations. Error: " + error);
            this.trainThread = null;
        })).start();
        return trainThread;
    }
    
    /**
     * Unterbricht das Training bei der nächsten Iteration.
     * 
     * Gibt true zurück, wenn ein Training-Thread lief und dieser zum Anhalten
     * aufgefordert wurde, sonst false.
     *
     * @return Erfolg
     */
    public boolean stopTraining() {
        return trainThread != null && (trainThreadStopped = true);
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
            layers[i].calcErrors(layers[i + 1]);
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
