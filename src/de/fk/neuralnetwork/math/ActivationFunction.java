package de.fk.neuralnetwork.math;

/**
 *
 * @author Felix
 */
public interface ActivationFunction {
    
    public static ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new LeakyReLU(), DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION = new Identity();
    
    public double apply(double in);
    
    public class Identity implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in;
        }

    }

    public class Sigmoid implements ActivationFunction {

        @Override
        public double apply(double in) {
            return 1.0 / (1.0 + (double) Math.exp(-in));
        }

    }
    
    public class ReLU implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in < 0.0 ? 0.0 : in;
        }

    }
    
    public class LeakyReLU implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in < 0.0 ? 0.01 * in : in;
        }

    }
    
}
