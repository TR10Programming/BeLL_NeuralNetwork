package de.fk.neuralnetwork;

import de.fk.neuralnetwork.data.ImageContainer;
import de.fk.neuralnetwork.data.Tester;
import de.fk.neuralnetwork.io.FileIO;
import de.fk.neuralnetwork.learning.Backpropagator;
import de.fk.neuralnetwork.training.ArrayTrainingSupplier;
import de.fk.neuralnetwork.training.LabeledImageTrainingSupplier;
import de.fk.neuralnetwork.training.TrainingExample;
import gui.MainFrame;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.json.JSONArray;
import org.json.JSONObject;


/**
 *
 * @author Felix
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            //mnistProblem();
            //xorProblem();
            mnistProblemVal();
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static final int NETS = 10, ITERATIONS = 200, ITERATIONS_WITHOUT_CHANGE = 5;
    public static final int[] NET_ARCHITECTURE = {784, 300, 100, 10};
    public static final double LEARNING_RATE = 0.3;
    public static final String TEMP_DIR = "mnist_val";
    public static final boolean AUTO_TRANSFORM = false;
    
    private static PrintStream outStream = null;
    
    private static void log(String msg) {
        System.out.print(msg);
        outStream.print(msg);
    }
    
    public static void mnistProblemVal() throws IOException {
        //Setup logging
        new File(TEMP_DIR).mkdirs();
        File logFile = new File(TEMP_DIR, "log.txt");
        if(!logFile.exists()) logFile.createNewFile();
        outStream = new PrintStream(new BufferedOutputStream(new FileOutputStream(logFile, true)), true);
        //Read sets
        log("Initializing...\n");
        ImageContainer.readFromArchive(new File("myset.sets"));
        NeuralNetwork[] nets = new NeuralNetwork[NETS];
        Backpropagator[] bps = new Backpropagator[NETS];
        LabeledImageTrainingSupplier[] suppliers = new LabeledImageTrainingSupplier[NETS];
        double[] bestvals = new double[NETS];
        Thread[] trainthreads = new Thread[NETS];
        
        System.out.print("Continue training? (y/n) ");
        Scanner inputScanner = new Scanner(System.in);
        int iteration = 0, iterationswithoutchange = 0;
        if("y".equals(inputScanner.next())) {
            System.out.print("Save String: ");
            JSONObject saveObj = new JSONObject(inputScanner.next());
            iteration = saveObj.getInt("i") + 1;
            iterationswithoutchange = saveObj.getInt("iwc") + 1;
            JSONArray jBestVals = saveObj.getJSONArray("bestvals");
            for(int i = 0; i < NETS; i++) bestvals[i] = jBestVals.getDouble(i);
            double learningRate = 0.3;
            for(int i = 0; i < iteration; i++) learningRate *= 0.995;
            //Load nets, init backpropagators, suppliers
            for(int i = 0; i < NETS; i++) {
                nets[i] = FileIO.read(new File(TEMP_DIR, "latest_" + i + ".jnet"));
                bps[i] = new Backpropagator(i, nets[i], learningRate, 0, 0);
                bps[i].setIteration(iteration - 1);
                suppliers[i] = new LabeledImageTrainingSupplier(ImageContainer::trainingSupplier, 28, 28, 10, AUTO_TRANSFORM);
                log("Loaded net #" + i + "\n");
            }
        } else {
            //Init nets, backpropagators, suppliers, arrays, initially save nets
            for(int i = 0; i < NETS; i++) {
                nets[i] = new NeuralNetwork(NET_ARCHITECTURE);
                bps[i] = new Backpropagator(i, nets[i], LEARNING_RATE, 0, 0);
                suppliers[i] = new LabeledImageTrainingSupplier(ImageContainer::trainingSupplier, 28, 28, 10, AUTO_TRANSFORM);
                bestvals[i] = Integer.MAX_VALUE;
                FileIO.write(new File(TEMP_DIR, "best_" + i + ".jnet"), nets[i], false);
                log("Initialized net #" + i + "\n");
            }
        }
        
        //Training
        for(; iteration < ITERATIONS && iterationswithoutchange < ITERATIONS_WITHOUT_CHANGE;
                iteration++, iterationswithoutchange++) {
            log("Iteration #" + iteration + ": Training all nets... (" + System.currentTimeMillis() + ", no change for " + iterationswithoutchange + " iterations)\n");
            //Train all n nets
            for(int i = 0; i < NETS; i++) trainthreads[i] = bps[i].train(suppliers[i], 1);
            //Wait for training, then test on validation set and, if better, save to file
            for(int i = 0; i < NETS; i++) {
                try {
                    trainthreads[i].join();
                } catch (InterruptedException ex) {
                    Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
                }
                log("Testing net #" + i + "... (" + System.currentTimeMillis() + ")");
                double valerror = Tester.testFromSet(nets[i], ImageContainer.Set.VALIDATION).getError();
                log("Err_val=" + valerror + "\n");
                if(valerror < bestvals[i]) {
                    bestvals[i] = valerror;
                    FileIO.write(new File(TEMP_DIR, "best_" + i + ".jnet"), nets[i], false);
                    iterationswithoutchange = 0;
                }
                FileIO.write(new File(TEMP_DIR, "latest_" + i + ".jnet"), nets[i], false);
            }
            JSONObject saveObject = new JSONObject();
            saveObject.put("i", iteration);
            saveObject.put("iwc", iterationswithoutchange);
            saveObject.put("bestvals", new JSONArray(bestvals));
            log(saveObject.toString() + "\n");
        }
        //Testing
        for(int net = 0; net < NETS; net++) {
            NeuralNetwork bestnet = FileIO.read(new File(TEMP_DIR, "best_" + net + ".jnet"));
            double testaccuracy = Tester.testFromSet(bestnet, ImageContainer.Set.TEST).getAccuracy();
            log("Net #" + net + ": Err_val(min)=" + bestvals[net] + " Test accuracy=" + testaccuracy + "\n");
        }
        log("Done!\n");
    }
    
    public static void mnistProblem() {
        try {
            ImageContainer.readFromMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 100, ImageContainer.FileFormat.MNIST, ImageContainer.Set.TRAINING);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        //for(int i = 0; i < 10; i++) System.out.println(ImageContainer.getImages().get(i).toString());
        NeuralNetwork nn = new NeuralNetwork(784, 10, 300);
        
        LabeledImageTrainingSupplier trainingSupplier = new LabeledImageTrainingSupplier(() -> ImageContainer.getImages(ImageContainer.Set.TRAINING), 28, 28, MainFrame.NUM_CLASSES, false);
        
        Backpropagator bp = new Backpropagator(0, nn, 0.03, 0, 0);
        FileOutputStream fos = null;
        try {
            //Logging
            fos = new FileOutputStream("errorlog.txt");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        bp.setLogStream(fos);
        try {
            Tester.testFromMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", nn, 10000, ImageContainer.FileFormat.MNIST);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        for(int i = 0; i < 5; i++) {
            bp.trainParallel(trainingSupplier, 100, 1, 100, true);
            //TODO Warten
            try {
                Tester.testFromMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", nn, 10000, ImageContainer.FileFormat.MNIST);
            } catch (IOException ex) {
                Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    public static void xorProblem() {
        NeuralNetwork nn = new NeuralNetwork(1, 3, 2, 1);
        
        ArrayTrainingSupplier trainingSupplier = new ArrayTrainingSupplier(2, 1);
        trainingSupplier.addTrainingExample(new TrainingExample(new double[]{0, 0}, new double[]{1}));
        trainingSupplier.addTrainingExample(new TrainingExample(new double[]{1, 0}, new double[]{0}));
        trainingSupplier.addTrainingExample(new TrainingExample(new double[]{0, 1}, new double[]{0}));
        trainingSupplier.addTrainingExample(new TrainingExample(new double[]{1, 1}, new double[]{1}));
        
        System.out.println("Zufälliges Netz generiert. Folgende Ausgaben macht das Netz für die Trainingsbeispiele:");
        
        trainingSupplier.getTrainingExamples().forEach((ex) -> {
            System.out.println(Arrays.toString(ex.getIn()) + " -> " + Arrays.toString(nn.trigger(ex.getIn()).getOutput()));
        });
        
        Backpropagator bp = new Backpropagator(0, nn, 3, 0, 0);
        FileOutputStream fos = null;
        try {
            //Logging
            fos = new FileOutputStream("errorlog.txt");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        bp.setLogStream(fos);
        bp.trainParallel(trainingSupplier, 100000, 1, 4, false);
        //TODO Warten
        
        if(fos != null) try {
            fos.flush();
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        System.out.println("Fertig trainiert. Folgende Ausgaben macht das Netz für die Trainingsbeispiele:");
        trainingSupplier.getTrainingExamples().forEach((ex) -> {
            System.out.println(Arrays.toString(ex.getIn()) + " -> " + Arrays.toString(nn.trigger(ex.getIn()).getOutput()));
        });
        
        System.out.println("Folgende Gewichte wurden ermittelt:");
        NeuralLayer[] layers = nn.getLayers();
        for(int l = 0; l < layers.length; l++) {
            Neuron[] neurons = layers[l].getNeurons();
            for(int n = 0; n < neurons.length; n++)
                System.out.println(l + "-" + n + ": " + ((neurons[n] instanceof BasicNeuron) ? Arrays.toString(((BasicNeuron) neurons[n]).getWeights()) : "Bias"));
        }
    }

}
