/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
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
    private int folds = 0;
    private double percent;
    private String confidence;
    private String testFilename;
            
    public MyWeka () {
        // do nothing
    }
    
    //****************** Arff and CSV File Reader ******************//
    
    public void readFileArff (String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        data = new Instances(reader);
        reader.close();
        
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);
    }
    
    public void readFileCsv (String filename) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(filename);
        data = source.getDataSet();
        
        // setting class attribute
        if (data.classIndex() == -1)
            data.setClassIndex(data.numAttributes() - 1);
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
    
    public void buildClustererSimpleKMeans () throws Exception {
        clusterer = new SimpleKMeans();
        clusterer.buildClusterer(train);
    }
    
    public void buildClustererMyKMeans () throws Exception {
        /*
        clusterer = new MyKMeans();
        clusterer.buildClusterer(train);
        */
    }
    
    /*
    di KMeans ga ada evaluate Model, TODO remove this
    
    public void evaluateModel () throws Exception {
        eval = new ClusterEvaluation();
        eval.evaluateClusterer(test);
        Evaluation eval2 = new Evaluation(test);
    }
    */
    
    //****************** Testing Option Setter ******************//
    /* TODO fix this section, belum sesuai dengan clustering */
    
    public void fullTraining () {
        train = data;
        test = data;
    }
    
    public void setTestCase (String filename) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        test = new Instances(reader);
        reader.close();
        
        // setting class attribute
        test.setClassIndex(data.numAttributes() - 1);
    }
    
    public void crossValidate (int folds) throws Exception {
        /*
        eval = new Evaluation(data);
        eval.crossValidateModel(classifier, data, folds, new Random(1));    
        */
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
        for (int i = 0; i < data.numAttributes()-1; i++) {
            if(Attribute.NUMERIC == data.attribute(i).type()){
                Double value = Double.valueOf(attributes[i]);
                newInstance.setValue(i, value);
            } else {
                newInstance.setValue(i, attributes[i]);
            }
        }
        
        double clsLabel = clusterer.clusterInstance(newInstance);
        newInstance.setClassValue(clsLabel);
        
        String result = data.classAttribute().value((int) clsLabel);
        
        System.out.println("Hasil Cluster Unseen Data: " + result);
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
        int option;
        
        do {
            System.out.println("\nClustering Algorithms:");
            System.out.println("1. WEKA's SimpleKMeans");
            System.out.println("2. MyKMeans");
            System.out.println("3. MyAgnes");
            System.out.print("> Your option: ");
            option = input.nextInt();
            input.nextLine();
        } while (option != 1 || option != 2 || option != 3);
        
        optCls = option;
    }
    
    //*** 4. Pilih Test Option ***//
    
    public void chooseTestOption () {
        Scanner input = new Scanner(System.in);
        int option;
        
        do {
            System.out.println("\nTest Options:");
            System.out.println("1. Full Training");
            System.out.println("2. Cross Validation");
            System.out.println("3. Presentage Split");
            System.out.println("4. Supplied Test Case");
            option = input.nextInt();
            input.nextLine();
        } while (option != 1 || option != 2 || option != 3 || option != 4);
        
        optTest = option;
        
        switch (option) {
            case 2:
                System.out.print("Masukkan nilai fold: ");
                folds = input.nextInt();
                input.nextLine();
                break;
            case 3:
                System.out.print("Masukkan nilai persen train data: ");
                percent = input.nextDouble();
                input.nextLine();
                break;
            case 4:
                System.out.print("Masukkan nilai persen train data: ");
                percent = input.nextDouble();
                input.nextLine();
                break;
            default:
                // do nothing
        }  
    }
    
    //*** 5. Start Classifying ***//
    
    public void startClassify () throws Exception {
        /*
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            train = data;
        } else if(optTest == 3) {
            splitPercentage(percent);
        } else if(optTest == 4) {
            setTestCase(testFilename);
            train = data;
        }

        if(optCls == 1) {
            buildClassifierID3();    
        } else if(optCls == 2) {
            buildClassifierJ48(confidence);
        } else if(optCls == 3) {
            buildClassifierNaiveBayes();
        } else if(optCls == 4) {
            buildClassifierMyID3();
        }
        
        if(optTest == 2){
            crossValidate(folds);
        } else {
            evaluateModel();
        }  
        
        //Print Result
        System.out.println(eval.toSummaryString("\nSummary\n======\n", false));   
        System.out.println(eval.toClassDetailsString("\nStatistic\n======\n"));
        System.out.println(eval.toMatrixString("\nConfusion Matrix\n======\n"));
        */
    }
    
    //*** 6. Classify Unseen Data ***//
    
    
    public void startClassifyUnseen () throws Exception {
        /*
        Scanner input = new Scanner(System.in);
        
        System.out.print("\nMasukkan nilai atribut (pisahkan dengan spasi): ");
        String in = input.nextLine();
        
        if(optTest == 1) {
            fullTraining();
        } else if(optTest == 2) {
            train = data;
        } else if(optTest == 3) {
            splitPercentage(percent);
        } else if(optTest == 4) {
            setTestCase(testFilename);
            train = data;
        }

        if(optCls == 1) {
            buildClassifierID3();    
        } else if(optCls == 2) {
            buildClassifierJ48(confidence);
        } else if(optCls == 3) {
            buildClassifierNaiveBayes();
        } else if(optCls == 4) {
            buildClassifierMyID3();
        }
        
        String[] attributes = in.split(" ");
        classifyUnseenData(attributes);  
        */
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
