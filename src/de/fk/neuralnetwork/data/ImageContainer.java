package de.fk.neuralnetwork.data;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Felix
 */
public class ImageContainer {
    
    public static final int MNIST_IMAGE_FILE_MAGIC_NUMBER = 2051,
            MNIST_LABEL_FILE_MAGIC_NUMBER = 2049;

    private static ArrayList<LabeledImage> images = new ArrayList<>();
    
    public static void clear() {
        images.clear();
    }
    
    public static List<LabeledImage> getImages() {
        return images;
    }
    
    public static void readFromMnist(String imageFile, String labelFile) throws IOException {
        //ImageFile einlesen und überprüfen
        ByteBuffer imageBytes = ByteBuffer.wrap(Files.readAllBytes(Paths.get(imageFile)));
        int magicNumber = imageBytes.getInt();
        if(magicNumber != MNIST_IMAGE_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei '" + imageFile + "' beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_IMAGE_FILE_MAGIC_NUMBER + ")");
        int numImg = imageBytes.getInt(),
                numRows = imageBytes.getInt(),
                numCols = imageBytes.getInt();
        //LabelFile einlesen und überprüfen
        ByteBuffer labelBytes = ByteBuffer.wrap(Files.readAllBytes(Paths.get(labelFile)));
        magicNumber = labelBytes.getInt();
        if(magicNumber != MNIST_LABEL_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei '" + labelFile + "' beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_LABEL_FILE_MAGIC_NUMBER + ")");
        int numLabels = labelBytes.getInt();
        if(numImg != numLabels)
            throw new IOException("Die beiden Dateien passen nicht zusammen: " + imageFile + " enthält " + numImg + " Bilder, aber " + labelFile + " enthält " + numLabels + " Labels.");
        //Bilder und Labels einlesen & zusammenfügen
        for(int i = 0; i < numImg; i++) {
            int[][] data = new int[numRows][numCols];
            for(int r = 0; r < numRows; r++)
                for(int c = 0; c < numCols; c++)
                    data[r][c] = imageBytes.get() & 0xFF; //unsigned
            int label = labelBytes.get() & 0xFF; //unsigned
            images.add(new LabeledImage(data, label));
        }
    }
    
}
