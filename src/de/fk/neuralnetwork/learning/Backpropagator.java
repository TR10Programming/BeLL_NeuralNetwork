package de.fk.neuralnetwork.learning;

import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.training.TrainingSupplier;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.NeuralNetworkState;
import de.fk.neuralnetwork.math.NeuralMath;
import de.fk.neuralnetwork.training.TrainingExample;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Felix
 */
public class Backpropagator {
    
    public static final int SAVE_EVERY_X_ITERATIONS = 10;

    private NeuralNetwork net;
    private double learningRate, regularizationRate, momentum;
    private Thread trainThread;
    private boolean trainThreadStopped;
    
    public Backpropagator(NeuralNetwork net, double learningRate, double regularizationRate, double momentum) {
        this.net = net;
        this.learningRate = learningRate;
        this.regularizationRate = regularizationRate;
        this.momentum = momentum;
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
     * @deprecated Ersetzt durch trainParallel
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
                    error += backpropStep(trainingSupplier.nextTrainingExample(), 1);
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
    
    private double error = 0.0;
    
    /**
     * Startet den Trainingsvorgang (Multi-Threading-fähig).
     *
     * @param trainingSupplier TrainingSupplier mit Trainingsbeispielen
     * @param iterations Wiederholungen (pro Wiederholung werden threadCount * examplesPerThread Trainigsbeispiele verwendet)
     * @param errorLoggerStream OutputStream auf dem Iteration und Fehlerrate durch Leerzeichen separiert zeilenweise ausgegeben werden
     * @param threadCount Anzahl der zu verwendenden Backpropagation-Threads
     * @param examplesPerThread Trainingsbeispiele, die jeder Thread für die Backpropagation verwenden soll (empfohlen: examplesCount / threadCount; sollte nicht größer sein, da sonst Beispiele von mehreren Threads gleichzeitig trainiert werden)
     * @param staticExamples true, wenn für jeden Thread dauerhaft dieselben Trainingsbeispiele verwendet werden sollen (nur bei einer konstanten, endlichen Anzahl an Trainingsbeispielen sinnvoll)
     * @return Haupttrainingsthread
     * @throws IllegalStateException Wenn bereits ein Trainingsthread läuft
     * @see Backpropagator#stopTraining() 
     */
    public Thread trainParallel(TrainingSupplier trainingSupplier, int iterations, OutputStream errorLoggerStream, int threadCount, int examplesPerThread, boolean staticExamples) throws IllegalStateException {
        if(trainThread != null) throw new IllegalStateException("Es wird bereits trainiert.");
        trainThreadStopped = false;
        (trainThread = new Thread(() -> {
        
            long startTime = System.currentTimeMillis();
            int exampleCount = trainingSupplier.getExampleCount();
            net.prepareParallelBackprop(threadCount);
            
            System.out.println("Training with " + threadCount + " threads with " + examplesPerThread + " training examples each.");
            TrainingExample[][] examples = new TrainingExample[threadCount][examplesPerThread];
            for(int iteration = 0; iteration < iterations; iteration++) {
                if(trainThreadStopped) break;
                //Trainingsbeispiele anfordern
                if(!(staticExamples && iteration > 0)) {
                    for(int t = 0; t < threadCount; t++) {
                        examples[t] = trainingSupplier.nextTrainingExamples(examplesPerThread);
                        //System.out.println("Example 1 for thread #" + t + " out: " + Arrays.toString(examples[t][0].getOut()) + " In: " + Arrays.toString(examples[t][0].getIn()));
                    }
                }
                //Trainingsthreads starten
                Thread[] trainingThreads = new Thread[threadCount];
                error = 0.0;
                for(int t = 0; t < threadCount; t++) {
                    final TrainingExample[] threadExamples = examples[t];
                    final int threadId = t;
                    (trainingThreads[t] = new Thread(() -> {
                        double threadError = 0.0;
                        for(TrainingExample threadExample : threadExamples) {
                            threadError += backpropStep(threadExample, threadId);
                        }
                        error += threadError;
                    }, "TrainingThread#" + t)).start();
                }
                //Auf Threads warten
                for(int t = 0; t < threadCount; t++) try {
                    trainingThreads[t].join();
                } catch (InterruptedException ex) {
                    Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
                }
                //Learn/Apply accumulators
                Arrays.stream(net.getLayers()).parallel().forEach(l -> l.accumulate(learningRate, regularizationRate, momentum));
                //Fehler berechnen
                error /= threadCount * examplesPerThread;
                //Logging
                try {
                    double time = (System.currentTimeMillis() - startTime) / 1000.0;
                    byte[] out = (iteration + " " + error + " " + time + "\r\n").getBytes("UTF-8");
                    if(errorLoggerStream != null) errorLoggerStream.write(out);
                    System.out.write(out);
                    /*if(iteration % SAVE_EVERY_X_ITERATIONS == 0) {
                        //Kopie speichern
                        ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(new File("300-100_" + iteration + ".net")));
                        oos.writeObject(net);
                        oos.close();
                    }*/
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
     * Führt einen Backprop-Schritt aus und gibt den Fehler zurück (die Gewichte
     * werden nicht geupdatet!)
     *
     * @param trainingExample
     * @param threadId ID des ausführenden Threads (bei Single-Threading 0)
     * @return
     */
    public double backpropStep(TrainingExample trainingExample, int threadId) {
        NeuralLayer[] layers = net.getLayers();
        //Sammeln der Trainingsdaten
        double[] input = trainingExample.getIn(), expectedOutput = trainingExample.getOut();
        //Aktivierungen berechnen
        NeuralNetworkState out = net.trigger(input);
        //System.out.println("Training for " + Arrays.toString(input) + " -> " + Arrays.toString(expectedOutput));
        //Output Layer
        NeuralLayer outputLayer = layers[layers.length - 1];
        //Berechne Errors & Error Deltas
        double[] activationsBefore = layers.length > 1 ? out.getLayerActivations(layers.length - 2) : NeuralMath.addBias(input),
                errors = NeuralMath.getErrors(out.getOutput(), expectedOutput),
                errorDeltas = outputLayer.getErrorDeltas(errors, activationsBefore);
        //Berechne Accumulators
        outputLayer.calcAccumulatorMatrices(errorDeltas, activationsBefore, threadId);
        //Hidden Layer
        for(int i = layers.length - 2; i >= 0; i--) {
            //Berechne Errors & Error Deltas
            activationsBefore = i > 0 ? out.getLayerActivations(i - 1) : NeuralMath.addBias(input);
            errors = layers[i].getErrors(layers[i + 1], errorDeltas);
            errorDeltas = layers[i].getErrorDeltas(errors, activationsBefore);
            //Berechne Accumulators
            layers[i].calcAccumulatorMatrices(errorDeltas, activationsBefore, threadId);
        }
        //Fehler berechnen
        return NeuralMath.getRegularizedError(out.getOutput(), expectedOutput, regularizationRate, net);
    }
    
}
