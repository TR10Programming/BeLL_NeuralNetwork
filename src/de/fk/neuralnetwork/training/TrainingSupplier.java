package de.fk.neuralnetwork.training;

import de.fk.neuralnetwork.math.NeuralMath;
import java.util.stream.Stream;

/**
 *
 * @author Felix
 */
public abstract class TrainingSupplier {
    
    private int features, classes;
    private boolean autoAddBias;
    
    public TrainingSupplier(int features, int classes) {
        this.features = features;
        this.classes = classes;
        this.autoAddBias = false;
    }

    public void setAutoAddBias(boolean autoAddBias) {
        this.autoAddBias = autoAddBias;
    }

    public boolean isAutoAddBias() {
        return autoAddBias;
    }

    public int getFeatures() {
        return features;
    }

    public void setFeatures(int features) {
        this.features = features;
    }

    public int getClasses() {
        return classes;
    }

    public void setClasses(int classes) {
        this.classes = classes;
    }
    
    public TrainingExample nextTrainingExample() {
        TrainingExample te = supplyTrainingExample();
        return autoAddBias ? NeuralMath.addBias(te) : te;
    }
    
    /**
     * Stellt das nächste Trainingsbeispiel bereit.
     *
     * @return TrainingExample
     */
    protected abstract TrainingExample supplyTrainingExample();
    
    /**
     * Setzt den TrainingSupplier auf das erste Trainingsbeispiel zurück.
     * 
     * (Bei nicht zufälligen Suppliern sollte so wieder die gleiche Sequenz
     * an Trainingsbeispielen zurückgegeben werden)
     *
     */
    public abstract void reset();
    
    /**
     * Gibt die Anzahl an Trainingsbeispielen oder -1 zurück, wenn es sich um
     * einen Generator handelt, der unendlich viele Trainingsbeispiele erzeugen
     * kann.
     *
     * @return Anzahl an Trainingsbeispielen
     */
    public abstract int getExampleCount();
    
}