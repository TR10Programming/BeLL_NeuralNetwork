package de.fk.neuralnetwork.data;

import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.math.NeuralMath;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

/**
 *
 * @author Felix
 */
public class Tester {
    
    public static final int MNIST_IMAGE_FILE_MAGIC_NUMBER = 2051,
            MNIST_LABEL_FILE_MAGIC_NUMBER = 2049;

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
    
    public static int validateMnistImageFile(String imageFile) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(imageFile));
        int mn = dis.readInt();
        if(mn != MNIST_IMAGE_FILE_MAGIC_NUMBER) {
            dis.close();
            return -1;
        }
        int numImg = dis.readInt();
        dis.close();
        return numImg;
    }
    
    public static int validateMnistLabelFile(String labelFile) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(labelFile));
        if(dis.readInt() != MNIST_LABEL_FILE_MAGIC_NUMBER) {
            dis.close();
            return -1;
        }
        int numLabels = dis.readInt();
        dis.close();
        return numLabels;
    }
    
    public static class TestResult {
        
        private double error, accuracy;

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
