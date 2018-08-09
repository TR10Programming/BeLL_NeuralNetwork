package de.fk.neuralnetwork.math;

import java.util.stream.IntStream;

/**
 *
 * @author Felix
 */
public interface LossFunction {

    public double apply(double[] predicted, double[] expected);
    public double[] derivative(double[] predicted, double[] expected);
    
    public class MSE implements LossFunction {

        @Override
        public double apply(double[] predicted, double[] expected) {
            return IntStream.range(0, predicted.length)
                    .mapToDouble(i -> Math.pow(predicted[i] - expected[i], 2))
                    .sum() / 2;
        }

        @Override
        public double[] derivative(double[] predicted, double[] expected) {
            return IntStream.range(0, predicted.length)
                    .mapToDouble(i -> predicted[i] - expected[i])
                    .toArray();
        }
        
    }
    
}
