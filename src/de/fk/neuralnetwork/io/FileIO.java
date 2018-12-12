package de.fk.neuralnetwork.io;

import de.fk.neuralnetwork.BasicNeuron;
import de.fk.neuralnetwork.BiasNeuron;
import de.fk.neuralnetwork.NeuralLayer;
import de.fk.neuralnetwork.NeuralNetwork;
import de.fk.neuralnetwork.Neuron;
import de.fk.neuralnetwork.math.ActivationFunction;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Zum Speichern und Öffnen eines neuronalen Netzes im JSON-Format.
 *
 * @author Felix
 */
public class FileIO {

    /**
     * Speichert ein neuronales Netz in einer Datei (*.jnet, *.jfnet).
     *
     * @param f Datei
     * @param net Neuronales Netz
     * @param prettyPrint true, wenn lesbares JSON erzeugt werden soll (*.jfnet)
     * @throws IOException Schreibfehler
     */
    public static final void write(File f, NeuralNetwork net, boolean prettyPrint) throws IOException {
        try (FileWriter writer = new FileWriter(f)) {
            JSONObject jnet = new JSONObject();
            jnet.put("in", net.getInputNeurons());
            jnet.put("inBias", net.isInputBias());
            JSONArray jlayers = new JSONArray();
            
            //Layers
            for(NeuralLayer layer : net.getLayers()) {
                JSONObject jlayer = new JSONObject();
                jlayer.put("act", layer.getActivationFunction().getId());
                jlayer.put("actargs", new JSONArray(layer.getActivationFunction().getArgs()));
                
                //Neurons
                JSONArray jneurons = new JSONArray();
                for(Neuron neuron : layer.getNeurons()) {
                    JSONObject jneuron = new JSONObject();
                    if(neuron instanceof BiasNeuron) jneuron.put("bias", true);
                    else if(neuron instanceof BasicNeuron) {
                        jneuron.put("bias", false);
                        jneuron.put("weights", new JSONArray(((BasicNeuron) neuron).getWeights()));
                    }
                    jneurons.put(jneuron);
                }
                
                jlayer.put("neurons", jneurons);
                jlayers.put(jlayer);
            }
            
            jnet.put("layers", jlayers);
            writer.write(prettyPrint ? jnet.toString(2) : jnet.toString());
            writer.flush();
        }
    }
    
    /**
     * Öffnet ein neuronales Netz aus einer Datei (*.jnet, *.jfnet).
     *
     * @param f Datei
     * @return Neuronales Netz
     * @throws IOException Lesefehler
     */
    public static final NeuralNetwork read(File f) throws IOException {
        JSONObject jnet = new JSONObject(new String(Files.readAllBytes(f.toPath())));
        JSONArray jlayers = jnet.getJSONArray("layers");
        
        //Layers
        NeuralLayer[] layers = new NeuralLayer[jlayers.length()];
        for(int ilayer = 0; ilayer < layers.length; ilayer++) {
            JSONObject jlayer = jlayers.getJSONObject(ilayer);
            int actid = jlayer.getInt("act");
            JSONArray jactargs = jlayer.getJSONArray("actargs");
            double[] actargs = new double[jactargs.length()];
            for(int iargs = 0; iargs < actargs.length; iargs++) actargs[iargs] = jactargs.getDouble(iargs);
            
            //Neurons
            JSONArray jneurons = jlayer.getJSONArray("neurons");
            Neuron[] neurons = new Neuron[jneurons.length()];
            for(int ineuron = 0; ineuron < neurons.length; ineuron++) {
                JSONObject jneuron = jneurons.getJSONObject(ineuron);
                if(jneuron.getBoolean("bias")) neurons[ineuron] = new BiasNeuron();
                else {
                    JSONArray jweights = jneuron.getJSONArray("weights");
                    double[] weights = new double[jweights.length()];
                    for(int iweight = 0; iweight < weights.length; iweight++) weights[iweight] = jweights.getDouble(iweight);
                    neurons[ineuron] = new BasicNeuron(weights);
                }
            }
            
            layers[ilayer] = new NeuralLayer(ActivationFunction.fromId(actid, actargs), neurons);
        }
        
        return new NeuralNetwork(jnet.getInt("in"), jnet.getBoolean("inBias"), layers);
    }
    
}
