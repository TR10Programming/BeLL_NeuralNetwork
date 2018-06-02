package de.fk.neuralnetwork.io;

import de.fk.neuralnetwork.BasicNeuron;
import de.fk.neuralnetwork.BiasNeuron;
import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.Neuron;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.stream.Collectors;

/**
 *
 * @author Felix
 */
public class PHPExport {
    
    private static final DecimalFormat DFORMAT = new DecimalFormat("#", DecimalFormatSymbols.getInstance(Locale.US));
    
    static {
        DFORMAT.setMaximumFractionDigits(16);
    }

    public static final void write(File f, NeuralNetwork net) throws IOException {
        try (FileWriter writer = new FileWriter(f)) {
            String cmds = "";
            int in = net.getInputNeurons();
            NeuralLayer[] layers = net.getLayers();
            for(int i = 1; i <= layers.length; i++) {
                cmds += "\n$l" + i + "=array();\n";
                Neuron[] neurons = layers[i-1].getNeurons();
                for(int j = 0; j < neurons.length; j++) {
                    cmds += "$l" + i + "[" + j + "]=";
                    if(neurons[j] instanceof BasicNeuron) cmds += "m($l" + (i-1) + ",array(" + join(((BasicNeuron) neurons[j]).getWeights()) + "));\n";
                    else if(neurons[j] instanceof BiasNeuron) cmds += "1;\n";
                    else cmds += "0;\n";
                }
                cmds += "unset($l" + (i-1) + ");\n";
            }
            cmds += "\nreturn $l" + layers.length + ";";
            
            //Jsfile einlesen
            String phpfile = new BufferedReader(new InputStreamReader(JSExport.class.getResourceAsStream("phpexport.php"))).lines().collect(Collectors.joining("\n"))
                    .replace("$INcmds;", cmds)
                    .replace("$INin;", "" + in);
            
            //Ausgeben
            writer.write(phpfile);
            writer.flush();
        }
    }
    
    private static String join(double[] arr) {
        if(arr.length == 0) return "";
        String str = DFORMAT.format(arr[0]) + "";
        for(int i = 1; i < arr.length; i++) str += "," + DFORMAT.format(arr[i]);
        return str;
    }
    
}
