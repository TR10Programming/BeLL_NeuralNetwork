package de.fk.neuralnetwork.learning;

import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.training.TrainingSupplier;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.NeuralNetworkState;
import de.fk.neuralnetwork.data.ImageContainer;
import de.fk.neuralnetwork.data.Tester;
import de.fk.neuralnetwork.math.NeuralMath;
import de.fk.neuralnetwork.training.TrainingExample;
import java.io.IOException;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.function.Consumer;
import java.util.logging.Level;
import java.util.logging.Logger;
import javafx.util.Pair;

/**
 *
 * @author Felix
 */
public class Backpropagator {
    
    public static final int SAVE_EVERY_X_ITERATIONS = 10;
    /*public static final double ADAPTIVE_LEARNING_RATE_DOWN_MIN = 0.7,
            ADAPTIVE_LEARNING_RATE_UP_MIN = 1.025,
            ADAPTIVE_LEARNING_RATE_DOWN_MAX = 0.9,
            ADAPTIVE_LEARNING_RATE_UP_MAX = 1.15;*/
    public static final double ADAPTIVE_LEARNING_RATE_DOWN_MIN = 0.995,
            ADAPTIVE_LEARNING_RATE_UP_MIN = 0.995,
            ADAPTIVE_LEARNING_RATE_DOWN_MAX = 0.995,
            ADAPTIVE_LEARNING_RATE_UP_MAX = 0.995;

    private int id;
    private NeuralNetwork net;
    private double learningRate, regularizationRate, momentum;
    private boolean stopped, training, adaptiveLREnabled, calcVaccuracy;
    private OutputStream debugStream = System.out, logStream = null;
    private Runnable learningRateUpdated = null;
    private Consumer<Pair<Double, Double>> trainingProgressUpdated = null;
    
    public Backpropagator(int id, NeuralNetwork net, double learningRate, double regularizationRate, double momentum) {
        this.id = id;
        this.net = net;
        this.learningRate = learningRate;
        this.regularizationRate = regularizationRate;
        this.momentum = momentum;
        this.stopped = false;
        this.training = false;
        this.adaptiveLREnabled = true;
        this.calcVaccuracy = false;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setNet(NeuralNetwork net) {
        this.net = net;
    }

    public void setAdaptiveLearningRateEnabled(boolean adaptiveLREnabled) {
        this.adaptiveLREnabled = adaptiveLREnabled;
    }

    public void setCalcVaccuracy(boolean calcVaccuracy) {
        this.calcVaccuracy = calcVaccuracy;
    }

    public boolean isCalcVaccuracy() {
        return calcVaccuracy;
    }

    public int getId() {
        return id;
    }

    public void setDebugStream(OutputStream debugStream) {
        this.debugStream = debugStream;
    }

    public void setLogStream(OutputStream logStream) {
        this.logStream = logStream;
    }

    public void setLearningRateUpdated(Runnable learningRateUpdated) {
        this.learningRateUpdated = learningRateUpdated;
    }

    public void setTrainingProgressUpdated(Consumer<Pair<Double, Double>> trainingProgressUpdated) {
        this.trainingProgressUpdated = trainingProgressUpdated;
    }
    
    private void debug(String msg) {
        if(debugStream != null) try {
            debugStream.write(("[BP#" + id + "] " + msg + "\n").getBytes("UTF-8"));
        } catch (IOException ex) {
            Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
            Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    private void log(String msg) {
        if(logStream != null) try {
            logStream.write((msg + "\r\n").getBytes("UTF-8"));
        } catch (IOException ex) {
            Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
            Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public double getVaccuracy() {
        return vaccuracy;
    }

    public int getIteration() {
        return iteration;
    }

    public void setIteration(int iteration) {
        this.iteration = iteration;
    }
    
    private double terror = 0, lastTError = 0, vaccuracy = 0;
    private int iteration = 0, tthresholdRow = 0;
    
    public Thread train(TrainingSupplier trainingSupplier, int iterations) throws IllegalStateException {
        if(training) throw new IllegalStateException("Es wird bereits trainiert.");
        stopped = false;
        training = true;
        Thread trainThread;
        (trainThread = new Thread(() -> {
            long startTime = System.currentTimeMillis();
            log("iteration time trainerror valaccuracy");
            int exampleCount = trainingSupplier.getExampleCount();
            int example;
            trainingSupplier.reset();
            debug("Transformation: " + ((System.currentTimeMillis() - startTime) / (double) exampleCount) + " ms/example");
            net.prepareParallelBackprop(1);
            debug("Training with " + exampleCount + " examples per iteration.");
            terror = 0.0;
            vaccuracy = 0.0;
            tthresholdRow = 0;
            NeuralLayer[] layers = net.getLayers();
            
            //Trainingsschleife
            int toIteration = iteration + iterations;
            for(; !stopped && iteration < toIteration; iteration++) {
                //Backpropagation; Alle Trainingsbeispiele ansehen
                debug("Starting iteration " + (iteration + 1));
                debug("Backpropagating...");
                long tempTime = System.currentTimeMillis();
                startTime = tempTime;
                for(example = 0; example < exampleCount; example++) {
                    TrainingExample ex = trainingSupplier.nextTrainingExample();
                    terror += NeuralMath.getError(backpropStepParallel(layers, ex), ex.getOut());
                    //Lernen/Gewichte updaten
                    for(NeuralLayer l : layers) l.accumulate(learningRate, regularizationRate, momentum);
                    if(stopped) break;
                }
                terror /= (double) exampleCount;
                if(calcVaccuracy) vaccuracy = Tester.testFromSet(net, ImageContainer.Set.VALIDATION).getAccuracy();
                
                debug("Done. Error: " + terror + ". Val Accuracy: " + vaccuracy + ". BP Time: " + (System.currentTimeMillis() - tempTime) + "ms.");
                
                //Adaptive Lernrate
                if(adaptiveLREnabled && lastTError > 0.0) {
                    if(lastTError > terror) learningRate *= Math.max(ADAPTIVE_LEARNING_RATE_UP_MIN, Math.min(ADAPTIVE_LEARNING_RATE_UP_MAX, lastTError / terror));
                    else learningRate *= Math.max(ADAPTIVE_LEARNING_RATE_DOWN_MIN, Math.min(ADAPTIVE_LEARNING_RATE_DOWN_MAX, lastTError / terror));
                    debug("Lernrate angepasst: " + learningRate + "\n");
                    if(learningRateUpdated != null) learningRateUpdated.run();
                }
                log(iteration + " " + (startTime / 1000) + " " + terror + " " + vaccuracy + "\r\n");
                if(trainingProgressUpdated != null) trainingProgressUpdated.accept(new Pair<>(terror, vaccuracy));
                lastTError = terror;
                terror = 0.0;
                if(stopped) break;
            }
            //Ende der Schleife
            debug("Trained for " + iteration + " iterations. Error: " + lastTError);
            if(logStream != null) try {
                logStream.close();
            } catch (IOException ex) { }
            this.training = false;
        })).start();
        trainThread.setUncaughtExceptionHandler((Thread t, Throwable e) -> {
            e.printStackTrace();
            training = false;
        });
        return trainThread;
    }
    
    private TrainingExample[][] pbpExamples;
    private CyclicBarrier pbpTrainingBarrier;
    private long pbpStartTime;
    
    protected class TrainingRunnable implements Runnable {
        
        private CyclicBarrier trainingBarrier;
        private int threadId;

        public TrainingRunnable(CyclicBarrier trainingBarrier, int threadId) {
            this.trainingBarrier = trainingBarrier;
            this.threadId = threadId;
        }

        @Override
        public void run() {
            while(training) {
                for(TrainingExample threadExample : pbpExamples[threadId]) {
                    backpropStep(threadExample, threadId);
                }
                try {
                    trainingBarrier.await();
                } catch (InterruptedException | BrokenBarrierException ex) {
                    Logger.getLogger(Backpropagator.class.getName()).log(Level.SEVERE, null, ex);
                    break;
                }
            }
            debug("Trainingsthread #" + threadId + " angehalten!");
        }
        
    }
    
    /**
     * Startet den Trainingsvorgang (Multi-Threading-fähig).
     *
     * @param trainingSupplier TrainingSupplier mit Trainingsbeispielen
     * @param iterations Wiederholungen (pro Wiederholung werden threadCount * examplesPerThread Trainigsbeispiele verwendet)
     * @param threadCount Anzahl der zu verwendenden Backpropagation-Threads
     * @param examplesPerThread Trainingsbeispiele, die jeder Thread für die Backpropagation verwenden soll (empfohlen: examplesCount / threadCount; sollte nicht größer sein, da sonst Beispiele von mehreren Threads gleichzeitig trainiert werden)
     * @param staticExamples true, wenn für jeden Thread dauerhaft dieselben Trainingsbeispiele verwendet werden sollen (nur bei einer konstanten, endlichen Anzahl an Trainingsbeispielen sinnvoll)
     * @throws IllegalStateException Wenn bereits trainiert wird
     * @see Backpropagator#stopTraining() 
     */
    public void trainParallel(TrainingSupplier trainingSupplier, int iterations, int threadCount, int examplesPerThread, boolean staticExamples) throws IllegalStateException {
        if(training) throw new IllegalStateException("Es wird bereits trainiert.");
        training = true;
        stopped = false;
        //Initialisieren
        log("iteration time trainerror valaccuracy");
        int exampleCount = trainingSupplier.getExampleCount(), fullTrainingCycle = exampleCount / (examplesPerThread * threadCount);
        net.prepareParallelBackprop(threadCount);
        pbpExamples = new TrainingExample[threadCount][examplesPerThread];
        terror = 0.0;
        lastTError = -1.0;
        vaccuracy = 0.0;
        iteration = 0;
        trainingSupplier.reset();
        TrainingExample[] originalTrainingExamples = trainingSupplier.originalTrainingExamples();
        //Trainingsbeispiele laden
        for(int t = 0; t < threadCount; t++) pbpExamples[t] = trainingSupplier.nextTrainingExamples(examplesPerThread);
        pbpStartTime = System.currentTimeMillis();
        //CyclicBarrier erstellen
        pbpTrainingBarrier = new CyclicBarrier(threadCount, () -> {
            iteration++;
            //Lernen/Gewichte updaten
            Arrays.stream(net.getLayers()).forEach(l -> l.accumulate(learningRate, regularizationRate, momentum));
            
            //Alle Beispiele angesehen
            if(iteration % fullTrainingCycle == 0) {
                //Forward Propagation wiederholen, um Training Error zu bestimmen
                //Der Trainingsfehler auf den transformierten Beispiele wäre wegen
                //der konkurrierenden Threads viel schwieriger zu ermitteln und
                //nicht aussagekräftiger
                lastTError = terror;
                terror = 0.0;
                debug("Determining error and accuracy...");
                for(TrainingExample originalTrainingExample : originalTrainingExamples) {
                    double[] out = net.triggerParallel(originalTrainingExample.getIn()).getOutput();
                    terror += NeuralMath.getRegularizedError(out, originalTrainingExample.getOut(), regularizationRate, net);
                }
                terror /= originalTrainingExamples.length;
                if(calcVaccuracy) vaccuracy = Tester.testFromSet(net, ImageContainer.Set.VALIDATION).getAccuracy();
                //Adaptive Lernrate
                if(adaptiveLREnabled && lastTError > 0.0) {
                    if(lastTError > terror) learningRate *= Math.max(ADAPTIVE_LEARNING_RATE_UP_MIN, Math.min(ADAPTIVE_LEARNING_RATE_UP_MAX, lastTError / terror));
                    else learningRate *= Math.max(ADAPTIVE_LEARNING_RATE_DOWN_MIN, Math.min(ADAPTIVE_LEARNING_RATE_DOWN_MAX, lastTError / terror));
                    debug("Lernrate angepasst: " + learningRate + "\n");
                    if(learningRateUpdated != null) learningRateUpdated.run();
                }
                //Logging
                log(iteration + ((System.currentTimeMillis() - pbpStartTime) / 1000) + " " + terror + " " + vaccuracy + "\r\n");
                pbpStartTime = System.currentTimeMillis();
                if(trainingProgressUpdated != null) trainingProgressUpdated.accept(new Pair<>(terror, vaccuracy));
            }
            //Überprüfen ob Training gestoppt
            if(stopped || iteration >= iterations) {
                debug("Trainiert für " + iterations + " Iterationen.");
                training = false;
            }
            //Neue Trainingsbeispiele laden
            if(!staticExamples)
                for(int t = 0; t < threadCount; t++)
                    pbpExamples[t] = trainingSupplier.nextTrainingExamples(examplesPerThread);
        });
        
        //Threads erstellen
        for(int t = 0; t < threadCount; t++) new Thread(new TrainingRunnable(pbpTrainingBarrier, t), "TrainingThread#" + t).start();
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
        return training && (stopped = true);
    }
    
    /**
     * Führt einen Backprop-Schritt aus und gibt die Netzausgabe zurück (die
     * Gewichte werden nicht geupdatet!).
     * 
     * Sollte nur für Batch Gradient Descent verwendet werden.
     *
     * @param trainingExample
     * @param threadId ID des ausführenden Threads (bei Single-Threading 0)
     * @return
     * @see Backpropagator#backpropStepParallel(de.fk.neuralnetwork.training.TrainingExample) Für Stochastic Gradient Descent
     */
    public double[] backpropStep(TrainingExample trainingExample, int threadId) {
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
        return out.getOutput();
    }
    
    /**
     * Führt einen Backprop-Schritt aus und gibt die Netzausgabe zurück (die
     * Gewichte werden nicht geupdatet!) (nutzt parallele Streams).
     * 
     * Sollte nur für Stochastic Gradient Descent verwendet werden.
     *
     * @param layers
     * @param trainingExample
     * @return Fehler E
     * @see Backpropagator#backpropStep(de.fk.neuralnetwork.training.TrainingExample, int) Für paralleles Lernen (Batch Gradient Descent)
     */
    public double[] backpropStepParallel(NeuralLayer[] layers, TrainingExample trainingExample) {
        //Sammeln der Trainingsdaten
        double[] input = trainingExample.getIn(), expectedOutput = trainingExample.getOut();
        //Aktivierungen berechnen
        NeuralNetworkState out = net.triggerParallel(input);
        //System.out.println("Training for " + Arrays.toString(input) + " -> " + Arrays.toString(expectedOutput));
        //Output Layer
        NeuralLayer outputLayer = layers[layers.length - 1];
        //Berechne Errors & Error Deltas
        double[] activationsBefore = layers.length > 1 ? out.getLayerActivations(layers.length - 2) : NeuralMath.addBias(input),
                errors = NeuralMath.getErrors(out.getOutput(), expectedOutput),
                errorDeltas = outputLayer.getErrorDeltas(errors, activationsBefore);
        //Berechne Accumulators
        outputLayer.calcAccumulatorMatrices(errorDeltas, activationsBefore, 0);
        //Hidden Layer
        for(int i = layers.length - 2; i >= 0; i--) {
            //Berechne Errors & Error Deltas
            activationsBefore = i > 0 ? out.getLayerActivations(i - 1) : NeuralMath.addBias(input);
            errors = layers[i].getErrorsParallel(layers[i + 1], errorDeltas);
            errorDeltas = layers[i].getErrorDeltas(errors, activationsBefore);
            //Berechne Accumulators
            layers[i].calcAccumulatorMatrices(errorDeltas, activationsBefore, 0);
        }
        //Fehler berechnen
        return out.getOutput();
    }
    
}
