/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import weka.core.Instances;

/**
 *
 * @author zulvafachrina
 */
public class TestMyAgnes {
    public static void main(String args[]) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader("data/weather.arff"));
        Instances data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
        
        MyAgnes agnes = new MyAgnes(data, 2, MyAgnes.SINGLE);
        for(int i=0; i< data.numAttributes(); i++) {
            double[][] ranges = agnes.getRanges();
            System.out.println(ranges[i][MyAgnes.RANGE_MIN] + " " + ranges[i][MyAgnes.RANGE_MAX] + " " + ranges[i][MyAgnes.RANGE_WIDTH]);
        }
        System.out.println();
        System.out.println("Eucledian Distance: " + agnes.euclideanDistance(data.instance(1), data.instance(3)));
        
    }
}
