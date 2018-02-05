package de.fk.neuralnetwork.math;

import java.io.Serializable;

/**
 *
 * @author Felix
 */
public interface ActivationFunction extends Serializable {
    
    public static ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new Sigmoid(), DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION = new Sigmoid();
    
    public static ActivationFunction fromId(int id, double... args) {
        switch(id) {
            case 0: return new Identity();
            case 1: return new Sigmoid();
            case 2: return new ReLU();
            case 3: return new LeakyReLU(args[0]);
            default: return null;
        }
    }
    
    public double apply(double in);
    public double derivative(double in);
    public int getId();
    public double[] getArgs();
    
    public class Identity implements ActivationFunction {

        @Override
        public double apply(double in) {
            return in;
        }

        @Override
        public double derivative(double in) {
            return 1;
        }

        @Override
        public int getId() {
            return 0;
        }

        @Override
        public double[] getArgs() {
            return new double[]{};
        }

    }

    public class Sigmoid implements ActivationFunction {
        
        private static final long serialVersionUID = 66258921105185908L;

        @Override
        public double apply(double in) {
            return 1.0 / (1.0 + (double) Math.exp(-in));
        }

        @Override
        public double derivative(double in) {
            double sig = this.apply(in);
            return sig * (1 - sig);
        }

        @Override
        public int getId() {
            return 1;
        }

        @Override
        public double[] getArgs() {
            return new double[]{};
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

        @Override
        public int getId() {
            return 2;
        }

        @Override
        public double[] getArgs() {
            return new double[]{};
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

        @Override
        public int getId() {
            return 3;
        }

        @Override
        public double[] getArgs() {
            return new double[]{leakiness};
        }

    }
    
}
