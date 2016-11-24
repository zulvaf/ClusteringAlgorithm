/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import weka.clusterers.NumberOfClustersRequestable;
import weka.clusterers.RandomizableClusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformationHandler;
import weka.core.WeightedInstancesHandler;
import weka.core.Utils;
import weka.core.FastVector;

/**
 *
 * @author TOSHIBA PC
 */
public class MyKMeans extends RandomizableClusterer {
    
    /*** Class Attributes ***/
    
    private int m_NumClusters = 2; // numbers of cluster to generate, default = 2
    private Instances m_ClusterCentroids; // holds the cluster centroids
    private boolean m_dontReplaceMissing = false; // Replace missing values globally?
    private int[] m_ClusterSizes; // The number of instances in each cluster
    private int m_MaxIterations = 500; // Maximum number of iterations to be executed
    private int m_Iterations; // Keep track of the number of iterations completed before convergence
    private double[] m_squaredErrors; // Holds the squared errors for all clusters
    protected DistanceFunction m_DistanceFunction = new EuclideanDistance(); // the distance function used
    protected boolean m_FastDistanceCalc = false; // whether to use fast calculation of distances (using a cut-off)
    
    
    /*** Constructor ***/
    public MyKMeans (int numClusters) {
        m_NumClusters = numClusters;
    }
    
    /*** Getter and Setter ***/
    
    public void setNumClusters (int n) throws Exception {
        if (n <= 0) {
            throw new Exception("Number of clusters must be > 0");
        }
        m_NumClusters = n;
    }
    
    public int getNumClusters () {
        return m_NumClusters;
    }
    
    @Override
    public int numberOfClusters() throws Exception {
        return m_NumClusters;
    }
    
    
    /*** KMeans Methods */
    
    @Override
    public void buildClusterer(Instances data) throws Exception {
        /* build the clusterer using the traning data */
        
        // check whether the clusterer can hold the data
        getCapabilities().testWithFail(data);
        
        // init variables
        Instances instances = new Instances(data); // copy data
        instances.setClassIndex(-1);
        m_Iterations = 0; // init number of iterations
        m_squaredErrors = new double [m_NumClusters]; // init squaredErrors
        int[] clusterAssignments = new int [instances.numInstances()]; // current cluster assignments
        int[] prevClusterAssignments = new int [instances.numInstances()]; // previous cluster assignments
        m_DistanceFunction.setInstances(instances); // set instance to the distance function
        
        // init cluster centroids
        FastVector attInfo = new FastVector(instances.numAttributes());
        for (int i = 0; i < instances.numAttributes(); i++) {
            attInfo.addElement(instances.attribute(i));
        }
        m_ClusterCentroids = new Instances(instances.relationName(), attInfo, m_NumClusters);
        
        // set initial cluster centroids
        Random RandomO = new Random(10);
        Instances seedCandidates = new Instances(data);
        for (int i = 0; i < m_NumClusters; i++) {
            int instanceIdx = RandomO.nextInt(seedCandidates.numInstances());
            m_ClusterCentroids.add(seedCandidates.instance(instanceIdx));
            seedCandidates.delete(instanceIdx);
        }
        
        // do the iterations until converged
        boolean converged = false;
        while (!converged) {
            // preparation: save the previous clusters
            if (m_Iterations >= 0) {
                prevClusterAssignments = clusterAssignments;
                clusterAssignments = new int [instances.numInstances()];
            }
            
            // reinit squared errors
            for (int i = 0; i < m_NumClusters; i++) {
                m_squaredErrors[i] = 0;
            }
           
            // assign instances to clusters
            for (int instancesIdx = 0; instancesIdx < data.numInstances(); instancesIdx++) {
                int minDistanceIdx = 0;
                double minDistance = 
                    m_DistanceFunction.distance(data.instance(instancesIdx),
                        m_ClusterCentroids.instance(0));
                
                if (m_NumClusters > 1) {                    
                    for (int clustersIdx = 1; clustersIdx < m_NumClusters; clustersIdx++) {
                        double distance = 
                            m_DistanceFunction.distance(data.instance(instancesIdx),
                                m_ClusterCentroids.instance(clustersIdx));
                        
                        if (distance < minDistance) {
                            minDistanceIdx = clustersIdx;
                            minDistance = distance;
                        }
                    }
                }
                
                clusterAssignments[instancesIdx] = minDistanceIdx;
                m_squaredErrors[minDistanceIdx] += minDistance * minDistance;
            }
            
            // check whether the new assignment is the same with the previous one
                // if same, then it is converged already
                // if not, iterate again (not converged)
            if (m_Iterations >= 0) {
                converged = true;
                int i = 0;
                while (converged && i < data.numInstances()) {
                    converged = clusterAssignments[i] == prevClusterAssignments[i];
                    i++;
                }
            }
            
            // update centroids
            int emptyClusterCount = 0;
            Instances[] tempI = new Instances[m_NumClusters];
            m_ClusterSizes = new int [m_NumClusters];
            
            for (int i = 0; i < m_NumClusters; i++) {
                tempI[i] = new Instances(instances, 0);
            }
            for (int i = 0; i < instances.numInstances(); i++) {
                tempI[clusterAssignments[i]].add(instances.instance(i));
            }
            for (int i = 0; i < m_NumClusters; i++) {
                if (tempI[i].numInstances() == 0) {
                    // empty cluster
                    emptyClusterCount++;
                } else {
                    moveCentroid(i, tempI[i]);
                    
                }
                m_ClusterSizes[i] = tempI[i].numInstances();
            }
            
            // update number of iterations
            m_Iterations++;
        }       
    }
    
    public int clusterInstance(Instance instance) throws Exception {
        /* returns the index of the best cluster for the instance */
        
        int minDistanceIdx = 0;
        double minDistance = 
            m_DistanceFunction.distance(instance,
                m_ClusterCentroids.instance(0));

        if (m_NumClusters > 1) {                    
            for (int clustersIdx = 1; clustersIdx < m_NumClusters; clustersIdx++) {
                double distance = 
                    m_DistanceFunction.distance(instance,
                        m_ClusterCentroids.instance(clustersIdx));

                if (distance < minDistance) {
                    minDistanceIdx = clustersIdx;
                    minDistance = distance;
                }
            }
        }
        return minDistanceIdx;
    }
    
    protected double[] moveCentroid(int centroidIndex, Instances members) {
        /*  move the centroid to the center of the cluster
            returns an array of the new center's attribute values
        */
        
        double[] vals = new double[members.numAttributes()];

        for (int j = 0; j < members.numAttributes(); j++) {					
            vals[j] = members.meanOrMode(j);
            m_ClusterCentroids.instance(centroidIndex).setValue(j, vals[j]);

        }
        return vals;
    }
    
    
    /*** Miscellaneous ***/
    
    public Capabilities getCapabilities () {
        /* returns the capabilities of the clusterer */
        
        Capabilities result = super.getCapabilities();
        result.disableAll();
        result.enable(Capability.NO_CLASS);
 
        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capability.NUMERIC_ATTRIBUTES);
        result.enable(Capability.MISSING_VALUES);
 
        return result;
    }
    
    public String toString () {
        /* returns string interpretation and summary of the clusterer */
        
        if (m_ClusterCentroids == null) {
            return "No clusterer built yet!";
        }
        
        StringBuffer temp = new StringBuffer();
        temp.append("\nMyKMeans\n======\n");
        
        temp.append("\nNumber of iterations: " + m_Iterations);
        temp.append("\nWithin cluster sum of squared errors: " + Utils.sum(m_squaredErrors));
        temp.append("\nMissing values globally replaced with mean/mode");
        
        temp.append("\n\nCluster centroids:\n");
        
        for (int i = 0; i < m_ClusterCentroids.numInstances(); i++) {
            temp.append("\n- Cluster " + i + " (" + m_ClusterSizes[i] + " data) -");
            for (int j = 0; j < m_ClusterCentroids.numAttributes(); j++) {
                temp.append("\n" + m_ClusterCentroids.instance(i).attribute(j).name());
                Attribute att = m_ClusterCentroids.instance(i).attribute(j);
                if (att.isNominal() || att.isString() || att.isDate()) {
                    temp.append(": " + m_ClusterCentroids.instance(i).stringValue(j));
                } else if (att.isNumeric()) {
                    temp.append(": " + m_ClusterCentroids.instance(i).value(j));
                }
            }
            temp.append("\n");
        }
        
        temp.append("\n\n");
        return temp.toString();
    }
  
}