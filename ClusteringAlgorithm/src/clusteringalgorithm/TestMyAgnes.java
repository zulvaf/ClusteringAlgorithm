/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Instances;

/**
 *
 * @author zulvafachrina
 */
public class TestMyAgnes {
    public static void main(String args[]) throws IOException, Exception{
        BufferedReader reader = new BufferedReader(new FileReader("data/weather.nominal.arff"));
        Instances data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
        
        Clusterer agnes = new MyAgnes(data, 2, MyAgnes.COMPLETE);
        agnes.buildClusterer(data);
        ClusterEvaluation eval = new ClusterEvaluation();
        eval.setClusterer(agnes);
        eval.evaluateClusterer(data);
        System.out.println("===== My Agnes Clustering =====\n");
        System.out.println(eval.clusterResultsToString());
    }
}
