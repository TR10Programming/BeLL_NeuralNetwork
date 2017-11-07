package de.fk.neuralnetwork;


/**
 *
 * @author Felix
 */
public class Main {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(1, 5, 2, 1);
        nn.trigger(new double[]{0, 0});
        nn.trigger(new double[]{1, 0});
        nn.trigger(new double[]{1, 1});
        nn.trigger(new double[]{0, 1});
    }

}
