package de.fk.neuralnetwork.math;

import java.io.Serializable;

/**
 *
 * @author Felix
 */
public interface ActivationFunction extends Serializable {
    
    public static ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new Sigmoid(), DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION = new Sigmoid();
    
    public double apply(double in);
    public double derivative(double in);
    
    public class Identity implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in;
        }

        @Override
        public double derivative(double in) {
            return 1;
        }

    }

    public class Sigmoid implements ActivationFunction {

        @Override
        public double apply(double in) {
            return 1.0 / (1.0 + (double) Math.exp(-in));
        }

        @Override
        public double derivative(double in) {
            double sig = this.apply(in);
            return sig * (1 - sig);
        }

    }
    
    public class ReLU implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in > 0.0 ? in : 0.0;
        }

        @Override
        public double derivative(double in) {
            return in > 0.0 ? 1.0 : 0.0;
        }

    }
    
    public class LeakyReLU implements ActivationFunction {
        
        private double leakiness;
        
        public LeakyReLU(double leakiness) {
            this.leakiness = leakiness;
        }

        @Override
        public double apply(double in) {
            return in < 0.0 ? leakiness * in : in;
        }

        @Override
        public double derivative(double in) {
            return in > 0.0 ? 1.0 : leakiness;
        }

    }
    
}
