package de.fk.neuralnetwork.math;

import java.io.Serializable;
import java.util.Arrays;

/**
 *
 * @author Felix
 */
public abstract class ActivationFunction implements Serializable {
    
    public static ActivationFunction DEFAULT_ACTIVATION_FUNCTION = new Sigmoid(), DEFAULT_OUTPUT_LAYER_ACTIVATION_FUNCTION = new Sigmoid();
    
    public static ActivationFunction fromId(int id, double... args) {
        switch(id) {
            case 0: return new Identity();
            case 1: return new Sigmoid();
            case 2: return new ReLU();
            case 3: return new LeakyReLU(args[0]);
            case 4: return new Softmax();
            default: return null;
        }
    }
    
    public abstract double apply(double in);
    public abstract double derivative(double in);
    public abstract boolean needsAllInputs();
    
    public double[] applyAll(double... in) {
        return Arrays.stream(in).map(this::apply).toArray();
    }
    
    public double[] derivativeAll(double... in) {
        return Arrays.stream(in).map(this::derivative).toArray();
    }
    
    public abstract int getId();
    public abstract double[] getArgs();
    
    public static class Identity extends ActivationFunction {

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

        @Override
        public double[] applyAll(double... in) {
            return in;
        }

        @Override
        public double[] derivativeAll(double... in) {
            double[] out = new double[in.length];
            Arrays.fill(out, 1);
            return out;
        }

        @Override
        public boolean needsAllInputs() {
            return false;
        }

    }

    public static class Sigmoid extends ActivationFunction {
        
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

        @Override
        public boolean needsAllInputs() {
            return false;
        }

    }
    
    public static class ReLU extends ActivationFunction {

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

        @Override
        public boolean needsAllInputs() {
            return false;
        }

    }
    
    public static class LeakyReLU extends ActivationFunction {
        
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

        @Override
        public boolean needsAllInputs() {
            return false;
        }

    }
    
    public static class Softmax extends ActivationFunction {

        @Override
        public double apply(double in) {
            throw new UnsupportedOperationException("Softmax Funktion nicht für Skalare zulässig.");
        }

        @Override
        public double derivative(double in) {
            throw new UnsupportedOperationException("Softmax Funktion nicht für Skalare zulässig.");
        }

        @Override
        public int getId() {
            return 4;
        }

        @Override
        public double[] getArgs() {
            return new double[]{};
        }

        @Override
        public double[] applyAll(double... in) {
            double max = Arrays.stream(in).max().getAsDouble();
            double[] out = Arrays.stream(in).map(inn -> Math.exp(inn - max)).toArray();
            double sum = Arrays.stream(out).sum();
            return Arrays.stream(out).map(outn -> outn / sum).toArray();
        }

        @Override
        public double[] derivativeAll(double... in) {
            double[] softmax = applyAll(in);
            double[] out = new double[in.length];
            for(int i = 0; i < in.length; i++) {
                for(int j = 0; j < in.length; j++) {
                    out[j] += i == j ? (softmax[i] * (1 - softmax[i])) : (-softmax[i] * softmax[j]);
                }
            }
            return out;
        }

        @Override
        public boolean needsAllInputs() {
            return true;
        }
        
    }
    
}
