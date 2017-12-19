package de.fk.neuralnetwork.training;

import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Felix
 */
public class ArrayTrainingSupplier extends TrainingSupplier {

    private final ArrayList<TrainingExample> trainingExamples = new ArrayList<>();
    private int iterator = 0;
    
    public ArrayTrainingSupplier(int features, int classes) {
        super(features, classes);
    }
    
    public void addTrainingExample(TrainingExample te) {
        trainingExamples.add(te);
    }
    
    public List<TrainingExample> getTrainingExamples() {
        return trainingExamples;
    }

    @Override
    protected TrainingExample supplyTrainingExample() {
        if(iterator >= trainingExamples.size()) iterator = 0;
        return trainingExamples.get(iterator++);
    }

    @Override
    public void reset() {
        iterator = 0;
    }

    @Override
    public int getExampleCount() {
        return trainingExamples.size();
    }

}
