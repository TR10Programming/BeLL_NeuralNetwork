package de.fk.neuralnetwork.data;

import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.stream.Collectors;
import javafx.util.Pair;

/**
 * Beinhaltet verschiedene Methoden zum Testen der Zuverlässigkeit eines
 * neuronalen Netzes.
 *
 * @author Felix
 */
public class Tester {
    
    /**
     * Die Magic Number eines MNIST Image Files.
     */
    public static final int MNIST_IMAGE_FILE_MAGIC_NUMBER = 2051;

    /**
     * Die Magic Number eines MNIST Label Files.
     */
    public static final int MNIST_LABEL_FILE_MAGIC_NUMBER = 2049;

    /**
     * Gibt eine Hashmap zurück, die allen vom neuronalen Netz falsch
     * klassifizierten LabeledImages ihren Wert der Fehlerfunktion zuweist.
     * Die Einträge sind nach absteigendem Fehler bzw. steigender Genauigkeit
     * geordnet.
     *
     * @param nn Netz, welches zum Klassifizieren verwendet werden soll
     * @param imgs Zu klassifizierende Bilder
     * @return HashMap
     */
    public static HashMap<LabeledImage, Double> findIncorrectlyClassified(NeuralNetwork nn, List<LabeledImage> imgs) {
        return imgs.parallelStream()
                .map(img -> new Pair<>(img, nn.trigger(NeuralMath.flatten(img.getData())).getOutput()))
                .filter(p -> NeuralMath.getPredictedLabel(p.getValue()) != p.getKey().getLabel())
                .map(p -> new Pair<>(p.getKey(), NeuralMath.getError(p.getValue(), NeuralMath.getOutputForLabel(p.getKey().getLabel(), p.getValue().length))))
                .sorted((o1, o2) -> Double.compare(o2.getValue(), o1.getValue()))
                .collect(Collectors.toMap(Pair::getKey, Pair::getValue, (e1, e2) -> e1, LinkedHashMap::new));
    }
    
    /**
     * Testet die Zuverlässigkeit eines neuronalen Netzes anhand von
     * MNIST-Daten.
     *
     * @param imageFile MNIST Image File
     * @param labelFile MNIST Label File
     * @param nn Zu testendes neuronales Netz
     * @param maxImages Maximale Anzahl einzulesender Bilder
     * @return Testergebnis mit Accuracy und Fehlerrate
     * @throws IOException Lesefehler
     * @see TestResult
     */
    public static TestResult testFromMnist(String imageFile, String labelFile, NeuralNetwork nn, int maxImages) throws IOException {
        double error = 0, accuracy = 0;
        
        ArrayList<LabeledImage> images = new ArrayList<>();
        //ImageFile einlesen und überprüfen
        ByteBuffer imageBytes = ByteBuffer.wrap(Files.readAllBytes(Paths.get(imageFile)));
        int magicNumber = imageBytes.getInt();
        if(magicNumber != MNIST_IMAGE_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei '" + imageFile + "' beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_IMAGE_FILE_MAGIC_NUMBER + ")");
        int numImg = Math.min(maxImages, imageBytes.getInt()),
                numRows = imageBytes.getInt(),
                numCols = imageBytes.getInt(),
                imgSize = numRows * numCols;
        //LabelFile einlesen und überprüfen
        ByteBuffer labelBytes = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)));
        magicNumber = labelBytes.getInt();
        if(magicNumber != MNIST_LABEL_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei '" + labelFile + "' beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_LABEL_FILE_MAGIC_NUMBER + ")");
        int numLabels = Math.min(maxImages, labelBytes.getInt());
        if(numImg != numLabels)
            throw new IOException("Die beiden Dateien passen nicht zusammen: " + imageFile + " enthält " + numImg + " Bilder, aber " + labelFile + " enthält " + numLabels + " Labels.");
        //Bilder und Labels einlesen & zusammenfügen
        for(int i = 0; i < numImg; i++) {
            double[] flatData = new double[numRows * numCols];
            for(int j = 0; j < imgSize; j++)
                    flatData[j] = (imageBytes.get() & 0xFF) / 255.0; //unsigned
            int label = labelBytes.get() & 0xFF; //unsigned
            double[] out = nn.trigger(flatData).getOutput();
            error += NeuralMath.getError(out, NeuralMath.getOutputForLabel(label, nn.getOutputLayer().getNeurons().length));
            accuracy += (NeuralMath.getPredictedLabel(out) == label) ? 1.0 : 0.0;
            //System.out.println("Predicted: " + NeuralMath.getPredictedLabel(out) + " Actual: " + label);
        }
        error /= (double) numImg;
        System.out.println(accuracy);
        accuracy /= (double) numImg;
        return new TestResult(error, accuracy);
    }
    
    /**
     * Beschreibt ein Testergebnis eines neuronalen Netzes.
     * 
     * Speichert die Accuracy und die Fehlerrate.
     *
     */
    public static class TestResult {
        
        private final double error, accuracy;

        public TestResult(double error, double accuracy) {
            this.error = error;
            this.accuracy = accuracy;
        }

        public double getError() {
            return error;
        }

        public double getAccuracy() {
            return accuracy;
        }
        
    }
    
}
