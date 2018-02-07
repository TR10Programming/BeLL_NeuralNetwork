package de.fk.neuralnetwork;

import de.fk.neuralnetwork.data.ImageContainer;
import de.fk.neuralnetwork.data.Tester;
import de.fk.neuralnetwork.learning.Backpropagator;
import de.fk.neuralnetwork.training.ArrayTrainingSupplier;
import de.fk.neuralnetwork.training.LabeledImageTrainingSupplier;
import de.fk.neuralnetwork.training.TrainingExample;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;


/**
 *
 * @author Felix
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        mnistProblem();
        //xorProblem();
    }
    
    public static void mnistProblem() {
        try {
            ImageContainer.readFromMnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte", 100);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        //for(int i = 0; i < 10; i++) System.out.println(ImageContainer.getImages().get(i).toString());
        NeuralNetwork nn = new NeuralNetwork(1, 300, 28*28, 10);
        
        LabeledImageTrainingSupplier trainingSupplier = new LabeledImageTrainingSupplier(ImageContainer::getImages, 28, 28, 10);
        
        Backpropagator bp = new Backpropagator(nn, 0.03, 0, 0);
        FileOutputStream fos = null;
        try {
            //Logging
            fos = new FileOutputStream("errorlog.txt");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        try {
            Tester.testFromMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", nn, 10000);
        } catch (IOException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        for(int i = 0; i < 5; i++) {
            bp.trainParallel(trainingSupplier, 100, fos, 1, 100, true);
            //TODO Warten
            try {
                Tester.testFromMnist("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", nn, 10000);
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
        
        Backpropagator bp = new Backpropagator(nn, 3, 0, 0);
        FileOutputStream fos = null;
        try {
            //Logging
            fos = new FileOutputStream("errorlog.txt");
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        bp.trainParallel(trainingSupplier, 100000, fos, 1, 4, false);
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
