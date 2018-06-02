package de.fk.neuralnetwork.training;

import de.fk.neuralnetwork.data.LabeledImage;
import de.fk.neuralnetwork.math.NeuralMath;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 *
 * @author Felix
 */
public class LabeledImageTrainingSupplier extends TrainingSupplier {
    
    private int imgWidth, imgHeight, classes, index;
    private List<LabeledImage> images;
    private Supplier<List<LabeledImage>> imageSupplier;
    private Random transformRdm;

    public LabeledImageTrainingSupplier(Supplier<List<LabeledImage>> imageSupplier, int imgWidth, int imgHeight, int classes) {
        super(imgWidth * imgHeight, classes);
        this.imgWidth = imgWidth;
        this.imgHeight = imgHeight;
        this.classes = classes;
        this.index = 0;
        this.imageSupplier = imageSupplier;
        this.images = imageSupplier.get();
        this.transformRdm = new Random();
    }

    @Override
    protected TrainingExample supplyTrainingExample() {
        if(index >= images.size()) reset();
        LabeledImage li = images.get(index++);
        return new TrainingExample(NeuralMath.flatten(li.getData()), NeuralMath.getOutputForLabel(li.getLabel(), classes));
    }

    @Override
    public void reset() {
        index = 0;
        //1. Neue Transformationen anwenden
        images = imageSupplier.get()
                .parallelStream()
                .map(limg -> limg.cloneAndTransform(transformRdm))
                .collect(Collectors.toList());
        //2. Shuffle
        Collections.shuffle(images, transformRdm);
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

    @Override
    protected TrainingExample[] supplyOriginalTrainingExamples() {
        return imageSupplier.get()
                .parallelStream()
                .map(li -> new TrainingExample(NeuralMath.flatten(li.getData()), NeuralMath.getOutputForLabel(li.getLabel(), classes)))
                .toArray(TrainingExample[]::new);
    }

}
