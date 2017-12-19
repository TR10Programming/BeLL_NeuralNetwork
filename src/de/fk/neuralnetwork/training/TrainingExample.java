package de.fk.neuralnetwork.training;

/**
 *
 * @author Felix
 */
public class TrainingExample {

    private double[] in, out;

    public TrainingExample(double[] in, double[] out) {
        this.in = in;
        this.out = out;
    }

    public double[] getIn() {
        return in;
    }

    public double[] getOut() {
        return out;
    }

    public void setIn(double[] in) {
        this.in = in;
    }

    public void setOut(double[] out) {
        this.out = out;
    }
    
}
