package de.fk.neuralnetwork.data;

import java.util.Random;

/**
 * Repräsentiert ein Graustufenbild mit einem Label.
 *
 * @author Felix
 */
public class LabeledImage {
    
    /**
     * Chance, dass Transformationen auf das Bild angewandt werden.
     *
     */
    public static final double GENERAL_TRANSFORMATION_CHANCE = 0.7;//70%
    
    /**
     * Maximale Drehung des Bildes beim Transformieren in beide Richtungen
     * (in Rad).
     *
     */
    public static final double ROTATION_BOUNDS = Math.PI / 5.0;//+-36°
    
    /**
     * Wahrscheinlichkeit, dass das Bild rotiert wird.
     *
     */
    public static final double ROTATION_CHANCE = 0.7;//70%
    
    /**
     * Maximale Vergrößerung/Verkleinerung beim Transformieren.
     *
     */
    public static final double SCALE_BOUNDS = 0.2;//+-20%
    
    /**
     * Wahrscheinlichkeit, dass das Bild skaliert wird.
     *
     */
    public static final double SCALE_CHANCE = 0.8;//80%
    
    /**
     * Maximale Verschiebung in x- und y-Richtung beim Transformieren.
     *
     */
    public static final int SHIFT_BOUNDS = 4;//+-4px
    
    /**
     * Wahrscheinlichkeit, dass das Bild verschoben wird.
     *
     */
    public static final double SHIFT_CHANCE = 0.9;//90%

    private double[][] data;
    private int label;
    
    /**
     * Erstellt ein neues LabeledImage.
     *
     * @param data Daten
     * @param label Klasse/Label
     */
    public LabeledImage(double[][] data, int label) {
        this.data = data;
        this.label = label;
    }

    public double[][] getData() {
        return data;
    }

    public int getLabel() {
        return label;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("== Label: ").append(label).append(" ==");
        for(int row = 0; row < data.length; row++) {
            for(int col = 0; col < data[row].length; col++) {
                double pixelVal = data[row][col];
                if(pixelVal == 0) sb.append(" ");
                else if(pixelVal < 1.0 / 3) sb.append(".");
                else if(pixelVal < 2.0 / 3) sb.append("x");
                else sb.append("X");
            }
            sb.append("\n");
        }
        return sb.toString();
    }
    
    /**
     * Gibt eine transformierte Kopie dieses Bildes zurück.
     * 
     * Folgende Transformationen werden angewendet: Rotation.
     *
     * @param r Random zum Bestimmen der Transformationen
     * @return Transformierte Kopie
     * @see LabeledImage#ROTATION_BOUNDS
     */
    public LabeledImage cloneAndTransform(Random r) {
        if(r.nextDouble() >= GENERAL_TRANSFORMATION_CHANCE) return this;
        //Daten transformieren
        double[][] newData = null;
        if(r.nextDouble() < SHIFT_CHANCE)
            newData = Preprocessing.shift(data, r.nextInt(2 * SHIFT_BOUNDS) - SHIFT_BOUNDS, r.nextInt(2 * SHIFT_BOUNDS) - SHIFT_BOUNDS);
        if(r.nextDouble() < SCALE_CHANCE)
            newData = Preprocessing.scale(newData == null ? data : newData, 1 + r.nextDouble() * 2 * SCALE_BOUNDS - SCALE_BOUNDS);
        if(r.nextDouble() < ROTATION_CHANCE)
            newData = Preprocessing.rotate(newData == null ? data : newData, r.nextDouble() * 2 * ROTATION_BOUNDS - ROTATION_BOUNDS);
        return new LabeledImage(newData == null ? data : newData, label);
    }
    
}
