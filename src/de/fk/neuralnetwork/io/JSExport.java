package de.fk.neuralnetwork.io;

import de.fk.neuralnetwork.BasicNeuron;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.Neuron;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.json.JSONArray;

/**
 *
 * @author Felix
 */
public class JSExport {

    public static final void write(File f, NeuralNetwork net) throws IOException {
        try (FileWriter writer = new FileWriter(f)) {
            JSONArray weights = new JSONArray();
            int in = net.getInputNeurons(), out = net.getOutputLayer().getNeurons().length;
            Stream.of(net.getLayers())
                    .map(layer -> {
                        JSONArray jlayer = new JSONArray();
                        for(Neuron neuron : layer.getNeurons())
                            if(neuron instanceof BasicNeuron) jlayer.put(new JSONArray(((BasicNeuron) neuron).getWeights()));
                        return jlayer;
                    })
                    .forEach(weights::put);
            
            //Jsfile einlesen
            String jsfile = new BufferedReader(new InputStreamReader(JSExport.class.getResourceAsStream("jsexport.js"))).lines().collect(Collectors.joining("\n"))
                    .replace("%weights%", weights.toString())
                    .replace("%in%", "" + in)
                    .replace("%out%", "" + out);
            
            //Ausgeben
            writer.write(jsfile);
            writer.flush();
        }
    }
    
}
