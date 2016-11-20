/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Locale;
import java.util.Vector;
import weka.clusterers.AbstractClusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author TOSHIBA PC
 */
public class MyAgnes extends AbstractClusterer {
    
    protected Instances trainData;
    protected int numClusters;
    protected int linkType;
    protected Node[] clusters;
    protected int[] clusterOfInstance;
    protected double[][] ranges;
    
    //static link type
    final static int SINGLE = 0;
    final static int COMPLETE = 1;
    
    //static for range index
    final static int RANGE_MIN = 0;
    final static int RANGE_MAX = 1;
    final static int RANGE_WIDTH = 2;


    
    class Node implements Serializable {
        Node m_left;
        Node m_right;
        Node m_parent;
        int m_iLeftInstance;
        int m_iRightInstance;
        double m_fLeftLength = 0;
        double m_fRightLength = 0;
        double m_fHeight = 0;
        public String toString(int attIndex) {
          NumberFormat nf = NumberFormat.getNumberInstance(new Locale("en","US"));
          DecimalFormat myFormatter = (DecimalFormat)nf;
          myFormatter.applyPattern("#.#####");

          if (m_left == null) {
            if (m_right == null) {
              return "(" + trainData.instance(m_iLeftInstance).stringValue(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              trainData.instance(m_iRightInstance).stringValue(attIndex) +":" + myFormatter.format(m_fRightLength) + ")";
            } else {
              return "(" + trainData.instance(m_iLeftInstance).stringValue(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              m_right.toString(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            }
          } else {
            if (m_right == null) {
              return "(" + m_left.toString(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              trainData.instance(m_iRightInstance).stringValue(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            } else {
              return "(" + m_left.toString(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +m_right.toString(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            }
          }
        }
        public String toString2(int attIndex) {
          NumberFormat nf = NumberFormat.getNumberInstance(new Locale("en","US"));
          DecimalFormat myFormatter = (DecimalFormat)nf;
          myFormatter.applyPattern("#.#####");

          if (m_left == null) {
            if (m_right == null) {
              return "(" + trainData.instance(m_iLeftInstance).value(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              trainData.instance(m_iRightInstance).value(attIndex) +":" + myFormatter.format(m_fRightLength) + ")";
            } else {
              return "(" + trainData.instance(m_iLeftInstance).value(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              m_right.toString2(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            }
          } else {
            if (m_right == null) {
              return "(" + m_left.toString2(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +
              trainData.instance(m_iRightInstance).value(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            } else {
              return "(" + m_left.toString2(attIndex) + ":" + myFormatter.format(m_fLeftLength) + "," +m_right.toString2(attIndex) + ":" + myFormatter.format(m_fRightLength) + ")";
            }
          }
        }
        void setHeight(double fHeight1, double fHeight2) {
          m_fHeight = fHeight1;
          if (m_left == null) {
            m_fLeftLength = fHeight1;
          } else {
            m_fLeftLength = fHeight1 - m_left.m_fHeight;
          }
          if (m_right == null) {
            m_fRightLength = fHeight2;
          } else {
            m_fRightLength = fHeight2 - m_right.m_fHeight;
          }
        }
        void setLength(double fLength1, double fLength2) {
          m_fLeftLength = fLength1;
          m_fRightLength = fLength2;
          m_fHeight = fLength1;
          if (m_left != null) {
            m_fHeight += m_left.m_fHeight;
          }
        }
    }
    
    public MyAgnes(Instances train, int clusters, int link){
        trainData = train;
        numClusters = clusters;
        linkType = link;
        
        initializeRanges();
    }
    
    public double[][] getRanges() {
        return ranges;
    }
        
    @Override
    public void buildClusterer(Instances data){
        clusters = new Node[numClusters];
        int numInstances = trainData.numInstances();
        clusterOfInstance = new int[numInstances];
        double[][] distanceMatrix;
     
        //initiate first cluster, every instance in one cluster
        Vector<Integer> [] nClusters = new Vector[numInstances];
        for(int i=0; i<numInstances; i++) {
            nClusters[i] = new Vector<Integer>();
            nClusters[i].add(i);
        }
        
        //initiate distanceMatrix using Eucledian Distance
        distanceMatrix = new double[numInstances][numInstances];
        for(int i=0; i<numInstances; i++){
            for(int j=i+1; j<numInstances; j++) {
                distanceMatrix[i][j] = euclideanDistance(trainData.instance(i), trainData.instance(j));
                distanceMatrix[j][i] = distanceMatrix[i][j];
            }
        }
        
        int iterateCluster = numInstances;
        Node[] clusterNodes = new Node[numInstances];
        while(iterateCluster > numClusters){
//            System.out.println("Iterate:" + iterateCluster);
//            for(int i=0; i<distanceMatrix.length; i++) {
//                for(int j=0; j<distanceMatrix[i].length; j++){
//                    System.out.print((float)distanceMatrix[i][j] + " ");
//                }
//                System.out.println();         
//            }
//            System.out.println();
            
            int[] minInstances = searchMin(distanceMatrix);
//            System.out.println(minInstances[0] + " " + minInstances[1]);
            Node node = mergeNodes(minInstances[0], minInstances[1], distanceMatrix, clusterNodes);
            
            clusterNodes[minInstances[0]] = node;
            
            nClusters[minInstances[0]].addAll(nClusters[minInstances[1]]);
            nClusters[minInstances[1]].removeAllElements();
            
            distanceMatrix = updateDistanceMatrix(distanceMatrix, minInstances[0], minInstances[1]);
            iterateCluster--;         
        }
          
        int iCurrent = 0;
        for(int i=0; i<numInstances; i++){
            if (nClusters[i].size() > 0) {
                for (int j = 0; j < nClusters[i].size(); j++) {
                    clusterOfInstance[nClusters[i].elementAt(j)] = iCurrent;
                }
                clusters[iCurrent] = clusterNodes[i];
                iCurrent++;
            }
        }
    }

    @Override
    public int numberOfClusters() {
        return numClusters;
    }
    
    
    
    public Node mergeNodes(int node1, int node2, double[][] matrix, Node[] clusterNodes){
        if (node1 > node2) {
          int temp = node1; node1 = node2; node2 = temp;
        }

        // track hierarchy
        Node node = new Node();
        if (clusterNodes[node1] == null) {
          node.m_iLeftInstance = node1;
        } else {
          node.m_left = clusterNodes[node1];
          clusterNodes[node1].m_parent = node;
        }
        if (clusterNodes[node2] == null) {
          node.m_iRightInstance = node2;
        } else {
          node.m_right = clusterNodes[node2];
          clusterNodes[node2].m_parent = node;
        }
        node.setHeight(matrix[node1][node2], matrix[node2][node1]);
        return node;    
    }
    
    public double[][] updateDistanceMatrix(double[][] matrix, int node1, int node2){ 
        double[][] new_matrix = matrix;
        if(linkType == SINGLE) {
            double[] cluster1 = matrix[node1];
            double[] cluster2 = matrix[node2];
            for(int i=0; i<cluster1.length; i++){
               if(cluster1[i] < cluster2[i]) {
                   new_matrix[node1][i] = cluster1[i];
                   new_matrix[i][node1] = cluster1[i];               
               } else {
                   new_matrix[node1][i] = cluster2[i];
                   new_matrix[i][node1] = cluster2[i];
               }
               new_matrix[node2][i] = Double.MAX_VALUE;
               new_matrix[i][node2] = Double.MAX_VALUE;     
            } 
            new_matrix[node1][node2] = Double.MAX_VALUE;
            new_matrix[node2][node1] = Double.MAX_VALUE;  
            
        } else {
            double[] cluster1 = matrix[node1];
            double[] cluster2 = matrix[node2];
            for(int i=0; i<cluster1.length; i++){
               if(cluster1[i] > cluster2[i]) {
                   new_matrix[node1][i] = cluster1[i];
                   new_matrix[node2][i] = cluster1[i];
               } else {
                   new_matrix[node1][i] = cluster2[i];
                   new_matrix[node2][i] = cluster2[i];
               }
            } 
        }

        return new_matrix;
    }
    
    public void initializeRanges(){
        ranges = new double[trainData.numAttributes()][3];
        if (trainData == null) {
            ranges = null;
        } else if(trainData.numAttributes() <= 0) {
            for (int i = 0; i < trainData.numAttributes(); i++) {
                ranges[i][RANGE_MIN] = Double.POSITIVE_INFINITY;
                ranges[i][RANGE_MAX] = -Double.POSITIVE_INFINITY;
                ranges[i][RANGE_WIDTH] = Double.POSITIVE_INFINITY;
            }
        } else {
            //initialize with first instance
            for (int i = 0; i < trainData.numAttributes(); i++) {
                if (!trainData.instance(0).isMissing(i)) {
                  ranges[i][RANGE_MIN] = trainData.instance(0).value(i);
                  ranges[i][RANGE_MAX] = trainData.instance(0).value(i);
                  ranges[i][RANGE_WIDTH] = 0.0;
                } else { // if value was missing
                  ranges[i][RANGE_MIN] = Double.POSITIVE_INFINITY;
                  ranges[i][RANGE_MAX] = -Double.POSITIVE_INFINITY;
                  ranges[i][RANGE_WIDTH] = Double.POSITIVE_INFINITY;
                }
            }

            //update ranges
            for(int i=1; i<trainData.numAttributes(); i++) {
                Instance instance = trainData.instance(i);
                for(int j=1; j < trainData.numAttributes(); j++){
                    double value = instance.value(j);
                    if(!instance.isMissing(j)) {
                       if (value < ranges[j][RANGE_MIN]) {
                        ranges[j][RANGE_MIN] = value;
                        ranges[j][RANGE_WIDTH] = ranges[j][RANGE_MAX] - ranges[j][RANGE_MIN];
                        if (value > ranges[j][RANGE_MAX]) { 
                          ranges[j][RANGE_MAX] = value;
                          ranges[j][RANGE_WIDTH] = ranges[j][RANGE_MAX] - ranges[j][RANGE_MIN];
                        }
                      } else {
                        if (value > ranges[j][RANGE_MAX]) {
                          ranges[j][RANGE_MAX] = value;
                          ranges[j][RANGE_WIDTH] = ranges[j][RANGE_MAX] - ranges[j][RANGE_MIN];
                        }
                      }
                    }
                }
            }
        }
    }
    public double euclideanDistance(Instance first, Instance second){
        //assumption : numAttribute first == numAttribute second, index first = index second
        
        double diff=0;
        
        for(int i=0; i < first.numAttributes(); i++) {
            if(i == trainData.classIndex())
                continue;
            diff+= Math.pow(calculateDifference(first.value(i),second.value(i),i),2);
        }
        
        return Math.sqrt(diff);
    }
    
    public double calculateDifference(double first, double second, int index) {
        if(trainData.attribute(index).type() == Attribute.NOMINAL) {
            if(Instance.isMissingValue(first) || Instance.isMissingValue(second) || first != second) {
                return 1;
            } else {
                return 0;
            }
        } else if(trainData.attribute(index).type() == Attribute.NUMERIC) {
            double diff;
            if(Instance.isMissingValue(first) && Instance.isMissingValue(second)) {
                diff = 1;
            } else if(Instance.isMissingValue(first)) {
                diff = normalize(second, index);
                if(diff < 0.5)
                    diff = 1.0 - diff;
            } else if(Instance.isMissingValue(second)) {
                diff = normalize(first, index);
                if(diff < 0.5)
                    diff = 1.0 - diff;
            } else {
                diff = normalize(first,index) - normalize(second,index);
            }
            return diff;
        } else {
            return 0;
        }
    }
    
    public double normalize(double val, int index) {
        if(Double.isNaN(ranges[index][RANGE_MIN]) || ranges[index][RANGE_WIDTH] == 0) {
            return 0;
        } else {
            return (val - ranges[index][RANGE_MIN])/ranges[index][RANGE_WIDTH];
        }
    }
    
    public int[] searchMin(double[][] matrix) {
        double min = matrix[0][1];
        int[] nodes = {0,1};
        for (int i=1; i< matrix.length; i++) {
            for(int j=i+1; j< matrix[i].length; j++) {
                if(matrix[i][j] < min){
                    min = matrix[i][j];
                    nodes[0] = i;
                    nodes[1] = j;
                }
            }
            
        }
        return nodes;
    }
    
    public void printModel(){
        System.out.println("=== Model and evaluation on training set ===\n");
        for(int i=0; i<numClusters; i++){
            System.out.println("Cluseter " + i);
            System.out.println(clusters[i].toString(trainData.classIndex()) + "\n");             
        }
    }
    
    
}
