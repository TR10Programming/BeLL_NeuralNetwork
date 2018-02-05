package de.fk.neuralnetwork.data;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Zum Lernen anhand von Bilddaten. Speichert alle Bilder vom Typ LabeledImage.
 *
 * @author Felix
 * @see LabeledImage
 */
public class ImageContainer {
    
    /**
     * Die Magic Number eines MNIST Image Files.
     */
    public static final int MNIST_IMAGE_FILE_MAGIC_NUMBER = 2051;

    /**
     * Die Magic Number eines MNIST Label Files.
     */
    public static final int MNIST_LABEL_FILE_MAGIC_NUMBER = 2049;

    private static ArrayList<LabeledImage> images = new ArrayList<>();
    
    /**
     * Entfernt alle geladenen Bilder aus dem Speicher.
     */
    public static void clear() {
        images.clear();
        System.gc();
    }
    
    /**
     * Gibt alle Bilder im Speicher als Liste zurück.
     *
     * @return
     */
    public static List<LabeledImage> getImages() {
        return images;
    }
    
    /**
     * Liest Bild- und Labeldaten aus MNIST-Dateien ein. Diese dürfen nicht
     * GZIP-komprimiert sein.
     *
     * @param imageFile Datei mit den Bilddaten (Anzahl x Höhe x Breite)
     * @param labelFile Datei mit den Labeldaten (Anzahl)
     * @param maxImages Anzahl an maximal einzulesenden Bildern
     * @throws IOException Lesefehler
     */
    public static void readFromMnist(String imageFile, String labelFile, int maxImages) throws IOException {
        //ImageFile einlesen und überprüfen
        ByteBuffer imageBytes = ByteBuffer.wrap(Files.readAllBytes(Paths.get(imageFile)));
        int magicNumber = imageBytes.getInt();
        if(magicNumber != MNIST_IMAGE_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei '" + imageFile + "' beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_IMAGE_FILE_MAGIC_NUMBER + ")");
        int numImg = Math.min(maxImages, imageBytes.getInt()),
                numRows = imageBytes.getInt(),
                numCols = imageBytes.getInt();
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
            double[][] data = new double[numRows][numCols];
            for(int r = 0; r < numRows; r++)
                for(int c = 0; c < numCols; c++)
                    data[r][c] = (imageBytes.get() & 0xFF) / 255.0; //unsigned
            int label = labelBytes.get() & 0xFF; //unsigned
            images.add(new LabeledImage(data, label));
        }
    }
    
    /**
     * Überprüft, ob es sich bei der übergebenen Datei um ein MNIST Image File
     * handelt und gibt ggf die Anzahl der enthaltenen Bilder zurück.
     *
     * @param imageFile Zu überprüfende Datei
     * @return Anzahl der enthaltenen Bilder oder -1, wenn es sich nicht um ein
     * MNIST Image File handelt.
     * @throws IOException Lesefehler
     */
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
    
    /**
     * Überprüft, ob es sich bei der übergebenen Datei um ein MNIST Label File
     * handelt und gibt ggf die Anzahl der enthaltenen Labels zurück.
     *
     * @param labelFile Zu überprüfende Datei
     * @return Anzahl der enthaltenen Labels oder -1, wenn es sich nicht um ein
     * MNIST Label File handelt.
     * @throws IOException Lesefehler
     */
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
    
}
