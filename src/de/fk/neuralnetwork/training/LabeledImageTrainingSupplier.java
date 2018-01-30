package de.fk.neuralnetwork.training;

import de.fk.neuralnetwork.data.LabeledImage;
import de.fk.neuralnetwork.math.NeuralMath;
import java.util.List;

/**
 *
 * @author Felix
 */
public class LabeledImageTrainingSupplier extends TrainingSupplier {
    
    private int imgWidth, imgHeight, classes, index;
    private List<LabeledImage> images;

    public LabeledImageTrainingSupplier(List<LabeledImage> images, int imgWidth, int imgHeight, int classes) {
        super(imgWidth * imgHeight, classes);
        this.imgWidth = imgWidth;
        this.imgHeight = imgHeight;
        this.classes = classes;
        this.index = 0;
        this.images = images;
    }

    @Override
    protected TrainingExample supplyTrainingExample() {
        if(index >= images.size()) index -= images.size();
        LabeledImage li = images.get(index++);
        double[][] data = li.getData();
        return new TrainingExample(NeuralMath.flatten(data), NeuralMath.getOutputForLabel(li.getLabel(), classes));
    }

    @Override
    public void reset() {
        index = 0;
    }

    @Override
    public int getExampleCount() {
        return images.size();
    }

    @Override
    protected TrainingExample[] supplyTrainingExamples(int count) {
        TrainingExample[] examples = new TrainingExample[count];
        for(int i = 0; i < count; i++) examples[i] = supplyTrainingExample();
        return examples;
    }

}
