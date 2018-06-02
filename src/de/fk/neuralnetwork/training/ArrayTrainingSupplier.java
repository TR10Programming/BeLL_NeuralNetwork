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
        if(iterator >= trainingExamples.size()) iterator -= trainingExamples.size();
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

    @Override
    protected TrainingExample[] supplyTrainingExamples(int count) {
        if(iterator >= trainingExamples.size()) iterator -= trainingExamples.size();
        TrainingExample[] examples = trainingExamples.subList(iterator, iterator + count).toArray(new TrainingExample[count]);
        iterator += count;
        return examples;
    }

    @Override
    protected TrainingExample[] supplyOriginalTrainingExamples() {
        return trainingExamples.toArray(new TrainingExample[trainingExamples.size()]);
    }

}
