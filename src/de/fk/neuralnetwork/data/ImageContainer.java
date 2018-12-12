package de.fk.neuralnetwork.data;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

/**
 * Zum Lernen anhand von Bilddaten. Speichert alle Bilder vom Typ LabeledImage.
 * Alle Bilder werden in einem von drei Sets gespeichert (Training Set, 
 * Validation Set, Test Set). Bilder können aus MNIST- oder EMNIST-Archiven
 * in ein Set eingelesen werden, umgekehrt können alle Sets in ein
 * MNIST-ähnliches ZIP-Archiv gespeichert werden.
 *
 * @author Felix
 * @see LabeledImage
 * @see Set
 */
public class ImageContainer {
    
    /**
     * Identifiziert den Datensatz, dem ein gelabeltes Bild angehört.
     *
     */
    public static enum Set {
        
        /**
         * Im Trainingsset werden alle Bilder gespeichert, von denen das Netz
         * direkt lernt. Der Backpropagation-Algorithmus nutzt ausschließlich
         * diese Bilder zum Anpassen der Gewichte und Biases. Meist das größte
         * Set (ca. 80% aller Bilder).
         *
         */
        TRAINING("train"),

        /**
         * Im Validierungsset werden alle Bilder gespeichert, die der
         * Lernalgorithmus zum Validieren der verwendeten Netzeigenschaften
         * oder Trainingsparameter (meist die Anzahl der Iterationen) nutzt.
         * Das Netz lernt so indirekt von diesem Set. Meist das kleinste Set
         * (ca. 5% aller Bilder), optional.
         */
        VALIDATION("val"),

        /**
         * Das Testset wird zum abschließenden Bewerten der Netzperformance
         * genutzt. Es ist wichtig, dass dieses Set vor Abschluss des Trainings-
         * vorgangs keine Verwendung gefunden hat, damit Anpassungen des Netzes
         * an das Testset vermieden werden. Ca. 15% aller Bilder.
         *
         */
        TEST("test");
        
        private final String alias;
        
        Set(String alias) {
            this.alias = alias;
        }

        /**
         * Gibt einen kurzen Bezeichner für das Set zurück.
         * 
         * Die Sets werden unter ihren Bezeichnern abgespeichert.
         *
         * @return
         */
        public String getAlias() {
            return alias;
        }
        
    }
    
    /**
     * Die Magic Number eines MNIST Image Files.
     */
    public static final int MNIST_IMAGE_FILE_MAGIC_NUMBER = 2051;

    /**
     * Die Magic Number eines MNIST Label Files.
     */
    public static final int MNIST_LABEL_FILE_MAGIC_NUMBER = 2049;

    private static ArrayList<LabeledImage> trainImages = new ArrayList<>(), valImages = new ArrayList<>(), testImages = new ArrayList<>();
    
    /**
     * Gibt das Dateiformat eines Datenarchivs an. Das MNIST- und EMNIST-Archiv
     * unterscheiden sich jeweils nur geringfügig in der Anordnung der Daten.
     *
     */
    public static enum FileFormat {
        MNIST, EMNIST;
    }
    
    /**
     * Entfernt alle geladenen Bilder aus einem Set.
     * 
     * @param set Set
     */
    public static void clearSet(Set set) {
        getImages(set).clear();
        System.gc();
    }
    
    /**
     * Entfernt alle geladenen Bilder aus dem Speicher.
     */
    public static void clearAll() {
        trainImages.clear();
        valImages.clear();
        testImages.clear();
        System.gc();
    }
    
    /**
     * Gibt alle Bilder aus einem Set als Liste zurück.
     *
     * @param set Set
     * @return
     */
    public static List<LabeledImage> getImages(Set set) {
        switch(set) {
            case TRAINING: return trainImages;
            case VALIDATION: return valImages;
            case TEST: return testImages;
            default: return null;
        }
    }
    
    /**
     * Gibt alle Bilder aus dem Trainingsset als Liste zurück.
     *
     * @return
     */
    public static List<LabeledImage> trainingSupplier() {
        return trainImages;
    }
    
    /**
     * Verschiebt die letzten x Bilder aus einem Set in das andere.
     *
     * @param from
     * @param to
     * @param amount
     */
    public static void moveImages(Set from, Set to, int amount) {
        List<LabeledImage> toMove = getImages(from).subList(getImages(from).size() - amount - 1, getImages(from).size() - 1);
        getImages(to).addAll(toMove);
        toMove.clear();
    }
    
    /**
     * Ordnet die Trainingsbeispiele in zufälliger Reihenfolge an.
     *
     */
    public static void shuffleTrainingImages() {
        Collections.shuffle(trainImages);
    }
    
    /**
     * Transformiert alle Trainingsbilder zufällig.
     * 
     * Achtung: Die Originalbilder gehen verloren.
     *
     * @param transformRdm Random zum Bestimmen der Stärke der Transformationen
     */
    public static void transformTrainingImages(Random transformRdm) {
        trainImages = trainImages
                .parallelStream()
                .map(limg -> limg.cloneAndTransform(transformRdm))
                .collect(Collectors.toCollection(ArrayList<LabeledImage>::new));
    }
    
    /**
     * Liest Bild- und Labeldaten aus MNIST-Dateien ein. Diese dürfen nicht
     * GZIP-komprimiert sein.
     *
     * @param imageFile Datei mit den Bilddaten (Anzahl x Höhe x Breite)
     * @param labelFile Datei mit den Labeldaten (Anzahl)
     * @param maxImages Anzahl an maximal einzulesenden Bildern
     * @param fileFormat
     * @param set
     * @throws IOException Lesefehler
     */
    public static void readFromMnist(String imageFile, String labelFile, int maxImages, FileFormat fileFormat, Set set) throws IOException {
        BufferedInputStream imageIn = new BufferedInputStream(new FileInputStream(imageFile));
        BufferedInputStream labelIn = new BufferedInputStream(new FileInputStream(labelFile));
        readFromMnist(imageIn, labelIn, maxImages, fileFormat, set);
    }
    
    /**
     * Liest Bild- und Labeldaten aus InputStreams ein.
     *
     * @param imgIn InputStream mit den Bilddaten (Anzahl x Höhe x Breite)
     * @param lblIn InputStream mit den Labeldaten (Anzahl)
     * @param maxImages Anzahl an maximal einzulesenden Bildern
     * @param fileFormat
     * @param set
     * @throws IOException Lesefehler
     */
    public static void readFromMnist(InputStream imgIn, InputStream lblIn, int maxImages, FileFormat fileFormat, Set set) throws IOException {
        DataInputStream dimgIn = new DataInputStream(imgIn);
        DataInputStream dlblIn = new DataInputStream(lblIn);
        //ImageFile einlesen und überprüfen
        int magicNumber = dimgIn.readInt();
        if(magicNumber != MNIST_IMAGE_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_IMAGE_FILE_MAGIC_NUMBER + ")");
        int numImg = Math.min(maxImages, dimgIn.readInt()),
                numRows = dimgIn.readInt(),
                numCols = dimgIn.readInt();
        List<LabeledImage> images = getImages(set);
        //LabelFile einlesen und überprüfen
        magicNumber = dlblIn.readInt();
        if(magicNumber != MNIST_LABEL_FILE_MAGIC_NUMBER)
            throw new IOException("Die Datei beginnt mit der Magic Number " + magicNumber + ". (Erwartet: " + MNIST_LABEL_FILE_MAGIC_NUMBER + ")");
        int numLabels = Math.min(maxImages, dlblIn.readInt());
        if(numImg != numLabels)
            throw new IOException("Die beiden Dateien passen nicht zusammen: Image File enthält " + numImg + " Bilder, aber Label File enthält " + numLabels + " Labels.");
        //Bilder und Labels einlesen & zusammenfügen
        switch(fileFormat) {
            case MNIST:
                for(int i = 0; i < numImg; i++) {
                    double[][] data = new double[numRows][numCols];
                    for(int r = 0; r < numRows; r++)
                        for(int c = 0; c < numCols; c++)
                            data[r][c] = (dimgIn.read() & 0xFF) / 255.0; //unsigned
                    int label = dlblIn.read() & 0xFF; //unsigned
                    images.add(new LabeledImage(data, label));
                }
                break;
            case EMNIST:
                for(int i = 0; i < numImg; i++) {
                    double[][] data = new double[numRows][numCols];
                    for(int c = 0; c < numCols; c++)
                        for(int r = 0; r < numRows; r++)
                            data[r][c] = (dimgIn.read() & 0xFF) / 255.0; //unsigned
                    int label = (dlblIn.read() & 0xFF) - 1; //unsigned
                    images.add(new LabeledImage(data, label));
                }
                break;
            default:
                throw new IllegalArgumentException("Ungültiges Dateiformat");
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
    
    /**
     * Speichert alle Sets in einem GZIP-Archiv. Auf der Root-Ebene werden drei
     * MNIST-Bildarchive und drei MNIST-Labelarchive angelegt, die die Bilder
     * und Labels jedes Sets enthalten.
     *
     * @param archive Speicherziel
     * @throws IOException Wenn ein Fehler beim Schreiben der Datei aufgetreten ist.
     */
    public static void saveToArchive(File archive) throws IOException {
        ZipOutputStream zos = new ZipOutputStream(new FileOutputStream(archive));
        DataOutputStream dos = new DataOutputStream(zos);
        //zos.setLevel(9);
        for(Set set : Set.values()) {
            ZipEntry imgEntry = new ZipEntry(set.getAlias() + "-images.idx3-ubyte");
            zos.putNextEntry(imgEntry);
            List<LabeledImage> imgList = getImages(set);
            byte[] labels = new byte[imgList.size()];
            //Image Header
            dos.writeInt(MNIST_IMAGE_FILE_MAGIC_NUMBER);
            dos.writeInt(imgList.size());
            dos.writeInt(28);
            dos.writeInt(28);
            //Image Data
            for(int i = 0; i < imgList.size(); i++) {
                LabeledImage img = imgList.get(i);
                labels[i] = (byte) img.getLabel();
                double[][] data = img.getData();
                for(int r = 0; r < 28; r++)
                    for(int c = 0; c < 28; c++)
                        dos.write((int) Math.round(data[r][c] * 255));
            }
            zos.closeEntry();
            ZipEntry lblEntry = new ZipEntry(set.getAlias() + "-labels.idx1-ubyte");
            zos.putNextEntry(lblEntry);
            //Label Header
            dos.writeInt(MNIST_LABEL_FILE_MAGIC_NUMBER);
            dos.writeInt(labels.length);
            //Label Data
            dos.write(labels);
            zos.closeEntry();
        }
        dos.flush();
        dos.close();
    }
    
    /**
     * Liest aus einem vollständigen Datenarchiv alle Sets ein.
     *
     * @param archive Speicherort des Archivs
     * @throws IOException Wenn ein Fehler beim Dateizugriff aufgetreten ist.
     * @see ImageContainer#saveToArchive(java.io.File) saveToArchive(..)
     */
    public static void readFromArchive(File archive) throws IOException {
        ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(archive)));
        DataInputStream dis = new DataInputStream(zis);
        HashMap<Set, ArrayList<LabeledImage>> images = new HashMap<>();
        ZipEntry entry;
        while((entry = zis.getNextEntry()) != null) {
            String ename = entry.getName().toLowerCase();
            Set eset = null;
            for(Set s : Set.values()) if(ename.startsWith(s.getAlias().toLowerCase())) {
                eset = s;
                break;
            }
            if(eset == null) continue;
            ArrayList<LabeledImage> imageList;
            if(images.containsKey(eset)) imageList = images.get(eset);
            else imageList = new ArrayList<>();
            int magicNumber = dis.readInt(),
                numEntries  = dis.readInt();
            if(imageList.size() < numEntries)
                for(int i = imageList.size(); i < numEntries; i++)
                    imageList.add(new LabeledImage(null, 0));
            if(magicNumber == MNIST_IMAGE_FILE_MAGIC_NUMBER) {
                //Images einlesen
                int numRows = dis.readInt(),
                    numCols = dis.readInt();
                for(int i = 0; i < numEntries; i++) {
                    double[][] data = new double[numRows][numCols];
                    for(int r = 0; r < numRows; r++)
                        for(int c = 0; c < numCols; c++)
                            data[r][c] = (dis.read() & 0xFF) / 255.0; //unsigned
                    imageList.get(i).setData(data);
                }
            } else if(magicNumber == MNIST_LABEL_FILE_MAGIC_NUMBER) {
                //Labels einlesen
                for(int i = 0; i < numEntries; i++) {
                    int label = dis.read() & 0xFF; //unsigned
                    imageList.get(i).setLabel(label);
                }
            }
            images.put(eset, imageList);
        }
        dis.close();
        //Bilder zu Sets hinzufügen
        images.entrySet().forEach(e -> getImages(e.getKey()).addAll(e.getValue()));
    }
    
}
