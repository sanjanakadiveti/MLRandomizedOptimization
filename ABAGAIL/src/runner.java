/**
 * Created by nidhi on 2/20/19.
 */
import dist.Distribution;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.example.NeuralNetworkWeightDistribution;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.*;
import shared.reader.CSVDataSetReader;
import shared.reader.DataSetReader;
import util.ABAGAILArrays;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.DoubleSummaryStatistics;

import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.TestTrainSplitFilter;
import shared.reader.ArffDataSetReader;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import func.nn.activation.LinearActivationFunction;



public class runner {
    static final int NUM_ITER = 10;
    static final int NUM_OF_SIMULATIONS = 1;
    static final int[] networkStrcuture = new int[] {31, 2, 1};



    private static Instance[] instances;

    private static int inputLayer, hiddenLayer, outputLayer = 1, trainingIterations;
    //private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet digits;

    public static void main(String[] args) {
      // Instance[] train_instances = initializeInstances(false);
      // Instance[] test_instances = initializeInstances(true);
      // DataSet set = new DataSet(train_instances);

        CSVDataSetReader data_all = new CSVDataSetReader(new File("").getAbsolutePath() + "/dataset_norm.csv");


        // read in the raw data
        DataSet data= null;
        DataSet train_data = null;
        DataSet test_data = null;
        // try {
        //     train_data = new DataSet(train_instances);
        //     // System.out.println(train_data.length());
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        // try {
        //     test_data = new DataSet(test_instances);
        // } catch (Exception e) {
        //     e.printStackTrace();
        // }
        try {
            data = data_all.read();
            // System.out.println(data.getLabelDataSet().getInstances().length);
        } catch (Exception e) {
            e.printStackTrace();
        }



        System.out.println("split");



//        instances = initializeInstances(path);
//        digits = new DataSet(instances);
//        inputLayer = instances[0].getData().size() - 1;
//        hiddenLayer = Integer.parseInt(args[0]);
//        trainingIterations = Integer.parseInt(args[1]);

        ArrayList<String> results = new ArrayList<>();
        if(data != null) {
            TestTrainSplitFilter ttsf = new TestTrainSplitFilter(75);
            ttsf.filter(data);
            train_data = ttsf.getTrainingSet();
            test_data = ttsf.getTestingSet();

            DataSet train_data_new;

            System.out.println("here");


                // TestTrainSplitFilter ttsf1 = new TestTrainSplitFilter(i);
                // ttsf1.filter(train_data);
                // train_data_new = ttsf1.getTrainingSet();
                // System.out.print(train_data_new.size() + " ");
            for (int x = 1; x <= 5000; x += 100){
              System.out.println(x);
              FeedForwardNetwork nnSA = runGA(train_data, x , 120, 0.3, 0.1);
              double test_fit = fitness(nnSA, test_data);
              System.out.print("TEST: " + test_fit);
              double train_fit = fitness(nnSA, train_data);
              System.out.println("TRAIN: " + train_fit);
            }




//                    FeedForwardNetwork nnRHC = runRHC(train_data_new, 300);
//
//                    //tempResults.append(String.format("%d,", sims));
//                    double[] test_fit = fitness(nnRHC, test_data);
//                    System.out.print(test_fit[0] +" " + test_fit[1] +" ");
//                    double[] train_fit = fitness(nnRHC, train_data_new);
//                    System.out.println(train_fit[0] +" " + train_fit[1]);
////
//
////                FeedForwardNetwork nnSA = runGA(train_data_new, 200);
////                double[] test_fit = fitness(nnSA, test_data);
////                System.out.print(test_fit[0] +" " + test_fit[1] +" ");
////                double[] train_fit = fitness(nnSA, train_data_new);
////                System.out.println(train_fit[0] +" " + train_fit[1]);
//
////                FeedForwardNetwork nnSA = runSA(train_data_new, 200, 0.55);
////                double[] test_fit = fitness(nnSA, test_data);
////                System.out.print(test_fit[0] +" " + test_fit[1] +" ");
////                double[] train_fit = fitness(nnSA, train_data_new);
////                System.out.println(train_fit[0] +" " + train_fit[1]);
//
//            }



//            for (int i = 0; i < train_data.size(); i++) {
//                train_data.get(i).setLabel(new Instance(train_data.get(i).getData().get(30)));
//                train_data.get(i).setData(train_data.get(i).getData().remove(30));
//            }
//
//            for (int i = 0; i < test_data.size(); i++) {
//                test_data.get(i).setLabel(new Instance(test_data.get(i).getData().get(30)));
//                test_data.get(i).setData(test_data.get(i).getData().remove(30));
//            }
            results.add("simulation number, RHC, SA, GA");

//            for(double mutate=0.05; mutate <= 1; mutate+=0.05){
//                System.out.print(mutate +" ");
//                FeedForwardNetwork nnSA = runGA(train_data, 200, 120, 0.3, mutate );
//                double[] test_fit = fitness(nnSA, test_data);
//                System.out.print(test_fit[0] +" " + test_fit[1] +" ");
//                double[] train_fit = fitness(nnSA, train_data);
//                System.out.println(train_fit[0] +" " + train_fit[1]);
//
//
//
//            }


//            for (int numIter=1; numIter<1000; numIter+=10) {
//                System.out.print(numIter+" ");
//
//                double recall_test;
//                double prec_test;
//                double acc_test;
//
//                double recall_train;
//                double prec_train;
//                double acc_train;
//
//                for (int sims = 0; sims <= NUM_OF_SIMULATIONS - 1; sims++) {
//                    long start = System.nanoTime();
//                    FeedForwardNetwork nnRHC = runRHC(train_data, numIter);
//                    StringBuffer tempResults = new StringBuffer();
//
//                    //tempResults.append(String.format("%d,", sims));
//                    double[] test_fit = fitness(nnRHC, test_data);
//                    System.out.print(test_fit[0] +" " + test_fit[1] +" ");
//                    double[] train_fit = fitness(nnRHC, train_data);
//                    System.out.println(train_fit[0] +" " + train_fit[1]);
//
//
//                    //System.out.println(tempResults);
//                    //tempResults.append(String.valueOf(fitness(nnMimic, data)));
//                    //tempResults.append(",");
//                    //tempResults.append(String.valueOf(fitness(nnSA, data)));
//                    //tempResults.append(",");
//                    //tempResults.append(String.valueOf(fitness(nnGA, data)));
//
//                    // End timer
//                    long end = System.nanoTime();
//
//                    // Print out results
//                    //System.out.println("----");
//                    //System.out.println("Time taken: " + (end - start) / Math.pow(10, 9) + " seconds.");
//                    results.add(tempResults.toString());
//                }
//
//                //System.out.println(avg_acc);
//                //System.out.println(avg_acc[0]);
//                //System.out.println(numIter + " " + sum_acc );
//
//            }
        }
        try{
            PrintWriter writer = new PrintWriter("results.csv", "UTF-8");
            for (String current: results) {
                writer.write((current + "\n").toCharArray());
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static FeedForwardNetwork runSA(DataSet data, int numIter, double cooling) {
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        FeedForwardNetwork network = factory.createClassificationNetwork(networkStrcuture);
        ErrorMeasure measure = new SumOfSquaresError();
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(data, network, measure);
        OptimizationAlgorithm o = new SimulatedAnnealing(10, cooling, nno);
        FixedIterationTrainer fit = new FixedIterationTrainer(o, numIter);
        fit.train();
        network.setWeights(o.getOptimal().getData());
        return network;
    }

    public static FeedForwardNetwork runGA(DataSet data, int numIter, int population, double mating, double mutation) {
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        FeedForwardNetwork network = factory.createClassificationNetwork(networkStrcuture);
        ErrorMeasure measure = new SumOfSquaresError();
        int toMate = (int)(population * mating);
        int toMutate = (int)(population *mutation);
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(data, network, measure);
        OptimizationAlgorithm o = new StandardGeneticAlgorithm(population, toMate, toMutate, nno);
        FixedIterationTrainer fit = new FixedIterationTrainer(o, numIter);
        fit.train();
        network.setWeights(o.getOptimal().getData());
        return network;
    }

    public static FeedForwardNetwork runRHC(DataSet data, int numIter) {
        FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
        FeedForwardNetwork network = factory.createClassificationNetwork(networkStrcuture, new LinearActivationFunction() );
        //System.out.print("made ir");
        ErrorMeasure measure = new SumOfSquaresError();
        NeuralNetworkOptimizationProblem nno = new NeuralNetworkOptimizationProblem(data, network, measure);
        OptimizationAlgorithm o = new RandomizedHillClimbing(nno);
        FixedIterationTrainer fit = new FixedIterationTrainer(o, numIter);
        fit.train();
        network.setWeights(o.getOptimal().getData());
        return network;
    }



    public static double train(ProbabilisticOptimizationProblem op) {
        Distribution distribution = op.getDistribution();
        int samples = 20;
        int tokeep = 10;
        Instance[] data = new Instance[samples];
        for (int i = 0; i < data.length; i++) {
            data[i] = distribution.sample(null);
        }
        double[] values = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            values[i] = op.value(data[i]);
        }
        double[] temp = new double[values.length];
        System.arraycopy(values, 0, temp, 0, temp.length);
        double cutoff = ABAGAILArrays.randomizedSelect(temp, temp.length - tokeep);
        int j = 0;
        Instance[] kept = new Instance[tokeep];
        for (int i = 0; i < data.length && j < kept.length; i++) {
            if (values[i] >= cutoff) {
                kept[j] = data[i];
                j++;
            }
        }
        distribution.estimate(new DataSet(kept));
        return cutoff;
    }


    /**
     * Fitness.
     *
     * @return the double
     */
    public static double fitness(FeedForwardNetwork network, DataSet data) {
        int correct = 0;
        int incorrect = 0;
        int true_pos = 0;
        int false_neg = 0;
        int true_neg = 0;
        int false_positive = 0;
        Instance[] patterns = data.getInstances();
        for (int i = 0; i < patterns.length; i++) {
            network.setInputValues(patterns[i].getData());
            network.run();

            double predicted = Double.parseDouble(patterns[i].getLabel().toString());
            // System.out.println(predicted);

            double actual = Double.parseDouble(network.getOutputValues().toString());
            if (Math.abs(predicted - actual) < 1) {
              correct++;
              // System.out.println(Math.abs(predicted - actual));
            } else {
              incorrect++;
            }


            // double correctLabel = data.get(i).getLabel().getContinuous();
            // double proposedLabel = network.getOutputValues().get(0);



            //System.out.println("Correct label: " + correctLabel);
            //System.out.println("Proposed label: " + proposedLabel);
            //System.out.println(correctLabel + " "+ proposedLabel);
            // if (correctLabel ==1 && proposedLabel >0.5 ){
            //     true_pos+=1;
            // } else if (correctLabel ==0 && proposedLabel <0.5 ){
            //     true_neg+=1;
            // } else if (correctLabel==0 && proposedLabel>0.5){
            //     false_positive+=1;
            // }
            // else if (correctLabel==1 && proposedLabel<0.5){
            //     false_neg+=1;
            // }
        }
        //System.out.println(true_pos);
        //System.out.println(true_pos);
        //System.out.println(false_positive);
        // double recall = (true_pos*1.0)/(true_pos*1.0+false_neg*1.0);
        // double precision = (true_pos*1.0)/(true_pos*1.0+false_positive*1.0);
        // double acc = (true_pos*1.0 + true_neg*1.0)/(true_pos*1.0+false_positive*1.0+false_neg*1.0+true_neg*1.0);
        // if(recall != recall){
        //     recall=0;
        //
        // }
        // if(precision != precision){
        //     precision=0;
        //
        // }
        // double[] A = {recall, precision, acc};
        double A = 100 - (1.0*correct/(correct+incorrect)*100.0);
        return A;
    }
    private static Instance[] initializeInstances(Boolean test) {
      double[][][] attributes = new double[7738][][];
      String name = "datasetTrain.csv";
      if (test) {
        System.out.println("here");
        attributes = new double[3317][][];
        name = "datasetTest.csv";
      }
      try {
        BufferedReader br = new BufferedReader(new FileReader(new File(name)));
        // adding now
        System.out.println(br.readLine());
        for (int i = 0; i < attributes.length; i++) {
          // System.out.println(i);
          Scanner scan = new Scanner(br.readLine());
          scan.useDelimiter(",");
          attributes[i] = new double[2][];
          attributes[i][0] = new double[31]; // bagSize attributes
          attributes[i][1] = new double[1];
          for(int j = 0; j < 31; j++)
            attributes[i][0][j] = Double.parseDouble(scan.next());
          attributes[i][1][0] = Double.parseDouble(scan.next());
        }
      } catch(Exception e) {
          e.printStackTrace();
      }
      Instance[] instances = new Instance[attributes.length];
      for(int i = 0; i < instances.length; i++) {
          instances[i] = new Instance(attributes[i][0]);
          instances[i].setLabel(new Instance(attributes[i][1][0] < .5 ? 0 : 1));
      }
      return instances;
    }

}
