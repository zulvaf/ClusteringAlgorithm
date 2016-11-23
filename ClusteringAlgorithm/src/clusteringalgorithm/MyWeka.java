/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Scanner;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author chairuniaulianusapati
 */

public class MyWeka {

    private Instances data;
    private Instances train;
    private Instances test;
    private Clusterer clusterer;
    private ClusterEvaluation eval;
    
    private int optCls = 1;
    private int optTest = 1;
    private double percent;
            
    public MyWeka () {
        // do nothing
    }
    
    //****************** Arff and CSV File Reader ******************//
    
    public void readFileArff (String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        reader.close();
    }
    
    public void readFileCsv (String filename) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
        data = source.getDataSet();
    }
    
    // ****************** Data Filter ****************** //
    
    public void removeAttribute (String attribute) throws Exception {
        String[] options = new String[2];
        options[0] = "-R";                                    
        options[1] = attribute;   
        
        Remove remove = new Remove();                         
        remove.setOptions(options);                           
        remove.setInputFormat(data);                          
        data = Filter.useFilter(data, remove);   
    }
    
    public void resample (String bias, String percent) throws Exception {
        String[] options = new String[4];
        options[0] = "-B";
        options[1] = bias;
        options[2] = "-Z";
        options[3] = percent;
        
        Resample sample = new Resample();
        sample.setOptions(options);
        sample.setInputFormat(data);
        data = Filter.useFilter(data, sample);     
    }
    
    //****************** Clusterer Builder ******************//
    
    public void buildClustererSimpleKMeans() throws Exception {
        clusterer = (SimpleKMeans) new SimpleKMeans();
        clusterer.buildClusterer(train);
    }
    
    public void buildClustererMyKMeans() throws Exception {
        clusterer = new MyKMeans();
        clusterer.buildClusterer(train);
    }

    public void buildClustererMyAgnes() throws Exception {
        clusterer = new MyAgnes(data, 2, MyAgnes.SINGLE);
        clusterer.buildClusterer(train);
    }
    
    public void evaluateModel () throws Exception {
        eval = new ClusterEvaluation();
        eval.setClusterer(clusterer);
        System.out.println("----- clusterer -----");
        System.out.println(clusterer);
        //eval.evaluateClusterer(test);
    }
    
    //****************** Testing Option Setter ******************//
    
    public void fullTraining () {
        train = data;
        test = data;
    }
 
    public void splitPercentage (double percent) {
        int trainSize = (int) Math.round(data.numInstances() * percent/ 100);
        int testSize = data.numInstances() - trainSize;
        
        train = new Instances(data, 0, trainSize);
        test = new Instances(data, trainSize, testSize);   
    }
    
    //****************** Unseen Data Clusterer ******************//
    
    public void clusterUnseenData (String[] attributes) throws Exception {
        Instance newInstance = new Instance(data.numAttributes());
        newInstance.setDataset(data);
        for (int i = 0; i < data.numAttributes(); i++) {
            if(Attribute.NUMERIC == data.attribute(i).type()){
                Double value = Double.valueOf(attributes[i]);
                newInstance.setValue(i, value);
            } else {
                newInstance.setValue(i, attributes[i]);
            }
        }
        
        double clsLabel = clusterer.clusterInstance(newInstance);        
        System.out.println("Data is in cluster: " +  (int) clsLabel);
    }
    
    //*** 1. Input Training Data ***//
    
    public void inputDataTrain () throws IOException, Exception {
        Scanner input = new Scanner(System.in);
        
        System.out.print("\n> Your training data input file: ");
        String filename = input.nextLine();
        
        if (filename.endsWith("arff")) {
            readFileArff(filename);
        } else if (filename.endsWith("csv")) {
            readFileCsv(filename);
        }
    }
    
    //*** 2. Filter Training Data ***//
    
    public void filtering () throws Exception {
        Scanner input = new Scanner(System.in);

        System.out.println("\nFiltering:");
        System.out.println("1. Remove Attribute");
        System.out.println("2. Resample");
        System.out.print("> Your option: ");
        
        int option = input.nextInt();
        input.nextLine();
        
        switch (option) {
            case 1: 
                System.out.print("> Index Atribute: ");
                String attribute = input.nextLine();
                removeAttribute(attribute);
                break;
            case 2:
                System.out.print("> Bias: ");
                String bias = input.nextLine();
                System.out.print("> Percentage: ");
                String percent = input.nextLine();
                resample(bias, percent);
                break;
            default:
                // do nothing
        }
    }
    
    //*** 3. Choose Clustering Algorithm ***//
    
    public void chooseClusteringAlgorithm () {
        Scanner input = new Scanner(System.in);
        int option = 0;
        
        while (option < 1 || option > 3) {
            System.out.println("\nClustering Algorithms:");
            System.out.println("1. WEKA's SimpleKMeans");
            System.out.println("2. MyKMeans");
            System.out.println("3. MyAgnes");
            System.out.print("> Your option: ");
            option = input.nextInt();
            input.nextLine();
        }
        
        optCls = option;
    }
    
    //*** 4. Pilih Test Option ***//
    
    public void chooseTestOption () {
        Scanner input = new Scanner(System.in);
        int option;
        
        do {
            System.out.println("\nTest Options:");
            System.out.println("1. Full Training");
            System.out.println("2. Precentage Split");
            option = input.nextInt();
            input.nextLine();
        } while (option < 1 || option > 2);
        
        optTest = option;
        
        switch (option) {
            case 2:
                System.out.print("Masukkan nilai persen train data: ");
                percent = input.nextDouble();
                input.nextLine();
                break;
            default:
                // do nothing
        }  
    }
    
    //*** 5. Build Clusterer ***//
    
    public void buildClusterer () throws Exception {
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            splitPercentage(percent);
        }

        if(optCls == 1) {
            buildClustererSimpleKMeans();    
        } else if(optCls == 2) {
            buildClustererMyKMeans();
        } else if(optCls == 3) {
            buildClustererMyAgnes();
        }
        
        evaluateModel();
        
        //Print Result
        System.out.println(eval.clusterResultsToString());
    }
    
    //*** 6. Cluster Unseen Data ***//
    
    public void startClusterUnseen () throws Exception {
        Scanner input = new Scanner(System.in);
        System.out.print("\nMasukkan nilai atribut (pisahkan dengan spasi): ");
        String in = input.nextLine();
        
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            splitPercentage(percent);
        }
        
        if(optCls == 1) {
            buildClustererSimpleKMeans();    
        } else if(optCls == 2) {
            buildClustererMyKMeans();
        } else if(optCls == 3) {
            buildClustererMyAgnes();
        }
        
        String[] attributes = in.split(" ");
        clusterUnseenData(attributes);
    }
    
    //*** 7. Print Data Summary ***//
    
    public void printDataSummary () {
        System.out.println("\nSummary\n======\n");
        System.out.println(data.toSummaryString());
    }
    
    //*** 8. Show Training Data ***//
    
    public void printTrainingData () {
        for (int i=0; i < data.numInstances(); i++) {
            System.out.print(i);
            System.out.print(": ");
            System.out.println(data.instance(i));
        }
    }
        
}
