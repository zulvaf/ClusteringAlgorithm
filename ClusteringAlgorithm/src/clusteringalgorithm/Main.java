/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package clusteringalgorithm;

import java.io.IOException;
import java.util.Scanner;
/**
 *
 * @author TOSHIBA PC
 */
public class Main {
    
    public static void printMainMenu () {
        System.out.println("\n-- MENU --");
        System.out.println("1. Input Training Data");
        System.out.println("2. Filter Training Data");
        System.out.println("3. Choose Clusterer");
        System.out.println("4. Configure Test Option");
        System.out.println("5. Build Clusterer");
        System.out.println("6. Cluster Unseen Data");
        System.out.println("7. Print Data Summary");
        System.out.println("8. Show Training Data");
        System.out.println("0. Quit");
    }
    
    public static void main (String[] args) throws IOException, Exception {
        Scanner input = new Scanner(System.in);
        MyWeka myWeka = new MyWeka();
        int option;
        
        System.out.println("---- Welcome To My Weka version 2.0 ----");
        
        printMainMenu();     
        System.out.print("\n> Your option: ");
        option = input.nextInt();
        input.nextLine();
        
        while(option != 0) {
            
            if (option == 1) {
                myWeka.inputDataTrain();
            } else if (option == 2) {
                myWeka.filtering();
            } else if (option == 3) {
                myWeka.chooseClusteringAlgorithm();
            } else if (option == 4) {
                myWeka.chooseTestOption();
            } else if (option == 5) {
                myWeka.startClassify();
            } else if (option == 6) {
                myWeka.startClassifyUnseen();
            } else if (option == 7) {
                myWeka.printDataSummary();
            } else if (option == 8) {
                myWeka.printTrainingData();
            }
            
            printMainMenu();     
            System.out.print("\n> Your option: ");
            option = input.nextInt();
            input.nextLine();
        }
    }
}
