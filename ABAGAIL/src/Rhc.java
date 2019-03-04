import shared.Instance;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.SumOfSquaresError;
import shared.filt.TestTrainSplitFilter;
import shared.reader.CSVDataSetReader;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import java.io.*;
import java.text.DecimalFormat;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import shared.DataSet;
import shared.FixedIterationTrainer;
import shared.DataSetDescription;
import shared.Instance;
import java.util.Arrays;


public class Rhc {
  private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
  private static FeedForwardNetwork network = factory.createClassificationNetwork(new int[] {31, 14, 14, 1});
  private static ErrorMeasure measure = new SumOfSquaresError();
  private static Instance[] train_instances = initializeInstances(false);
  private static Instance[] test_instances = initializeInstances(true);
  private static DataSet set = new DataSet(train_instances);;
  private static DecimalFormat df = new DecimalFormat("0.000");

  public static void main(String[] args) throws IOException {
    // experiment1(6000);
    NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
    OptimizationAlgorithm oa = new SimulatedAnnealing(1E9, 0.3, nnop);
    FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
    double start = System.nanoTime();
    fit.train();
    double end = System.nanoTime();
    double trainingTime = end - start;
    trainingTime /= Math.pow(10,9);
    System.out.println(trainingTime);
    // experiment1
    // experiment11();



    //experiment8(train_instances, test_instances);
    // experiment3();

  }

  private static void experiment1() {
    double[] testErr = new double[20];
    double[] trainErr = new double[20];
    double[] testTime = new double[20];
    double[] trainTime = new double[20];
    int curr = 0;
    for (int i = 100; i <= 2000; i+=100) {
      NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
      OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
      FixedIterationTrainer fit = new FixedIterationTrainer(oa, i);
      double s = System.nanoTime();
      fit.train();
      trainTime[curr] = System.nanoTime() - s;
      Instance optimalInstance = oa.getOptimal();
      network.setWeights(optimalInstance.getData());

      double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
      double predicted, actual;
      start = System.nanoTime();
      for(int j = 0; j < test_instances.length; j++) {
        network.setInputValues(test_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(test_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(10,9);
      testErr[curr] = 100 - (correct/(correct+incorrect)*100);
      System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      testTime[curr] = testingTime;

      correct = 0;
      incorrect = 0;
      for(int j = 0; j < train_instances.length; j++) {
        network.setInputValues(train_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(train_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      // System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      trainErr[curr] = (100 - (correct/(correct+incorrect)*100));
      System.out.println("Train Error: " + df.format(trainErr[curr++]));
    }
    System.out.println("Train Errors");
    for(double x : trainErr) System.out.println(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.println(y + ",");
    System.out.println("Train Time");
    for(double x : trainTime) System.out.println(x + ",");
    System.out.println("Test Time");
    for(double x : testTime) System.out.println(x + ",");
  }


  // RHC RANDOM RESTARTS
  private static void experiment2() {
    int[] numberOfRestarts = {1, 2, 3, 5, 8, 10, 15, 20};
    double[] testErr = new double[8];
    double[] trainErr = new double[8];
    double[] testTime = new double[8];
    double[] trainTime = new double[8];
    int curr = 0;
    for (int numberOfRestart : numberOfRestarts) {
        System.out.println("Training multiple NNs with RHC and numberOfRestarts=" + numberOfRestart);
        double bestError = 100000;
        OptimizationAlgorithm bestoa = null;
        double start = System.nanoTime(), end = 0, trainingTime = 0, testingTime = 0;
        for (int i = 0; i < numberOfRestart; i++) {
          NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
          OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
          FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
          fit.train();
          double error = 0;
          double correct = 0;
          double incorrect = 0;
          for(int j = 0; j < train_instances.length; j++) {
            network.setInputValues(train_instances[j].getData());
            network.run();
            Instance output = train_instances[j].getLabel(), example = new Instance(network.getOutputValues());
            example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
            error += measure.value(output, example);
            incorrect += Math.abs(Double.parseDouble(output.toString()) - Double.parseDouble(example.getLabel().toString())) < 0.5 ? 0 : 1;
          }
          if (((1.0*incorrect)/train_instances.length*100) < bestError) {
              bestError = ((1.0*incorrect)/train_instances.length*100);
              bestoa = oa;
          }
        }
        if (bestoa == null) throw new Error("bestoa not initialized");
        Instance optimalInstance = bestoa.getOptimal();
        network.setWeights(optimalInstance.getData());
        double correct = 0, incorrect = 0;
        double predicted, actual;
        for(int j = 0; j < test_instances.length; j++) {
          network.setInputValues(test_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(test_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        System.out.println("Train Error: " + df.format(bestError));
        trainErr[curr] = bestError;
        System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        testErr[curr] = 100 - (correct/(correct+incorrect)*100);
    }
    System.out.println("Train Errors");
    for(double x : trainErr) System.out.print(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.print(y + ",");
  }

  // Simulated Annealing - temperatures
  private static void experiment3() {
    double[] temperatures = {1E6, 1E7, 1E8, 1E9, 1E10, 1E11, 1E12};
    double[] trainErr = new double[7];
    double[] testErr = new double[7];
    int curr = 0;
    for (double temp : temperatures) {
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new SimulatedAnnealing(temp, .75, nnop);
        FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
        fit.train();
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());
        double correct = 0;
        double incorrect = 0;
        double predicted, actual;
        for(int j = 0; j < test_instances.length; j++) {
          network.setInputValues(test_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(test_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        testErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));

        correct = 0;
        incorrect = 0;
        for(int j = 0; j < train_instances.length; j++) {
          network.setInputValues(train_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(train_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        // System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        trainErr[curr] = (100 - (correct/(correct+incorrect)*100));
        System.out.println("Train Error: " + df.format(trainErr[curr++]));
      }
      System.out.println("Train Errors");
      for(double x : trainErr) System.out.print(x + ",");
      System.out.println("Test Errors");
      for(double y : testErr) System.out.print(y + ",");
  }

  // SA - Cooling
  private static void experiment4() {
    double[] trainErr = new double[3];
    double[] testErr = new double[3];
    int curr = 0;
    for (double i = 0.7; i < 1; i += 0.1) {
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new SimulatedAnnealing(1E9, i, nnop);
        FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
        fit.train();
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());
        double correct = 0;
        double incorrect = 0;
        double predicted, actual;
        for(int j = 0; j < test_instances.length; j++) {
          network.setInputValues(test_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(test_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        testErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));

        correct = 0;
        incorrect = 0;
        for(int j = 0; j < train_instances.length; j++) {
          network.setInputValues(train_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(train_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        // System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        trainErr[curr] = (100 - (correct/(correct+incorrect)*100));
        System.out.println("Train Error: " + df.format(trainErr[curr++]));
      }
      System.out.println("Train Errors");
      for(double x : trainErr) System.out.print(x + ",");
      System.out.println("Test Errors");
      for(double y : testErr) System.out.print(y + ",");
  }

  private static void experiment5() {
    double[] testErr = new double[20];
    double[] trainErr = new double[20];
    double[] testTime = new double[20];
    double[] trainTime = new double[20];
    int curr = 0;
    for (int i = 100; i <= 2000; i+=100) {
      NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
      OptimizationAlgorithm oa = new SimulatedAnnealing(1E9, 0.3, nnop);
      FixedIterationTrainer fit = new FixedIterationTrainer(oa, i);
      double s = System.nanoTime();
      fit.train();
      trainTime[curr] = System.nanoTime() - s;
      Instance optimalInstance = oa.getOptimal();
      network.setWeights(optimalInstance.getData());

      double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
      double predicted, actual;
      start = System.nanoTime();
      for(int j = 0; j < test_instances.length; j++) {
        network.setInputValues(test_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(test_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(10,9);
      testErr[curr] = 100 - (correct/(correct+incorrect)*100);
      System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      testTime[curr] = testingTime;

      correct = 0;
      incorrect = 0;
      for(int j = 0; j < train_instances.length; j++) {
        network.setInputValues(train_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(train_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      // System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      trainErr[curr] = (100 - (correct/(correct+incorrect)*100));
      System.out.println("Train Error: " + df.format(trainErr[curr++]));
    }
    System.out.println("Train Errors");
    System.out.println("");
    for(double x : trainErr) System.out.print(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.print(y + ",");
    System.out.println("Train Time");
    for(double x : trainTime) System.out.print(x + ",");
    System.out.println("Test Time");
    for(double x : testTime) System.out.print(x + ",");
    System.out.println("");
  }

  private static void experiment8() {
      double[] testErr = new double[8];
      double[] trainErr = new double[8];
      double[] testTime = new double[8];
      double[] trainTime = new double[8];
      set = new DataSet(train_instances);
      int curr = 0;
      int[] popSizes = {50, 100, 150, 200, 250, 300, 350, 400};

      for (int size : popSizes) {
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new StandardGeneticAlgorithm(size, 50, 50, nnop);
        FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
        System.out.println("here");
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        trainErr[curr] = trainingTime;
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < test_instances.length; j++) {
          network.setInputValues(test_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(test_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        testErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        testTime[curr] = testingTime;

        correct = 0;
        incorrect = 0;
        for(int j = 0; j < train_instances.length; j++) {
          network.setInputValues(train_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(train_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        trainErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        trainTime[curr++] = testingTime;
    }
    for(double x : trainErr) System.out.println(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.println(y + ",");
  }

  private static void experiment9() {
      double[] testErr = new double[4];
      double[] trainErr = new double[4];
      double[] testTime = new double[4];
      double[] trainTime = new double[4];
      set = new DataSet(train_instances);
      int curr = 0;
      int[] tomate = {25, 50, 75, 100};

      for (int size : tomate) {
        NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
        OptimizationAlgorithm oa = new StandardGeneticAlgorithm(200, 50, size, nnop);
        FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
        System.out.println("here");
        double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
        fit.train();
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);
        trainErr[curr] = trainingTime;
        Instance optimalInstance = oa.getOptimal();
        network.setWeights(optimalInstance.getData());

        double predicted, actual;
        start = System.nanoTime();
        for(int j = 0; j < test_instances.length; j++) {
          network.setInputValues(test_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(test_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        end = System.nanoTime();
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        testErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        testTime[curr] = testingTime;

        correct = 0;
        incorrect = 0;
        for(int j = 0; j < train_instances.length; j++) {
          network.setInputValues(train_instances[j].getData());
          network.run();
          predicted = Double.parseDouble(train_instances[j].getLabel().toString());
          actual = Double.parseDouble(network.getOutputValues().toString());
          double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
        }
        testingTime = end - start;
        testingTime /= Math.pow(10,9);
        trainErr[curr] = 100 - (correct/(correct+incorrect)*100);
        System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
        trainTime[curr++] = testingTime;
    }
    for(double x : trainErr) System.out.println(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.println(y + ",");
  }

  private static void experiment10(Instance[] train_instances, Instance[] test_instances) {
    double[] testErr = new double[20];
    double[] trainErr = new double[20];
    double[] testTime = new double[20];
    double[] trainTime = new double[20];
    set = new DataSet(train_instances);
    FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
    FeedForwardNetwork network = factory.createClassificationNetwork(new int[] {31, 14, 14, 1});
    int curr = 0;
    for (int i = 100; i <= 2000; i+=100) {
      NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(set, network, measure);
      OptimizationAlgorithm oa = new StandardGeneticAlgorithm(100, 25, 50, nnop);
      FixedIterationTrainer fit = new FixedIterationTrainer(oa, i);
      fit.train();
      Instance optimalInstance = oa.getOptimal();
      network.setWeights(optimalInstance.getData());

      double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
      double predicted, actual;
      start = System.nanoTime();
      for(int j = 0; j < test_instances.length; j++) {
        network.setInputValues(test_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(test_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(10,9);
      testErr[curr] = 100 - (correct/(correct+incorrect)*100);
      System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      testTime[curr] = testingTime;

      correct = 0;
      incorrect = 0;
      start = System.nanoTime();
      for(int j = 0; j < train_instances.length; j++) {
        network.setInputValues(train_instances[j].getData());
        network.run();
        predicted = Double.parseDouble(train_instances[j].getLabel().toString());
        actual = Double.parseDouble(network.getOutputValues().toString());
        double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
      }
      end = System.nanoTime();
      testingTime = end - start;
      testingTime /= Math.pow(10,9);
      trainErr[curr] = 100 - (correct/(correct+incorrect)*100);
      System.out.println("Train Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
      trainTime[curr++] = testingTime;

    }
    System.out.println("Train Errors");
    for(double x : trainErr) System.out.println(x + ",");
    System.out.println("Test Errors");
    for(double y : testErr) System.out.println(y + ",");
    System.out.println("Train Time");
    for(double x : trainTime) System.out.println(x + ",");
    System.out.println("Test Time");
    for(double x : testTime) System.out.println(x + ",");
  }
//comparing algorithms
  private static void experiment11() {
      double[] datasetPercentages = {0.05, 0.1, 0.3, 0.5, 0.7, 0.9};
      int curr = 0;
      double[] testErr = new double[6];
      double[] testTime = new double[6];
      double[] trainTime = new double[6];
      for (double datasetPercentage : datasetPercentages) {
          // Set up dataset.
          int lastIndex = (int)(train_instances.length * datasetPercentage);
          Instance[] curr_train_instances = Arrays.copyOfRange(train_instances, 0, lastIndex);
          DataSet currSet = new DataSet(curr_train_instances);
          System.out.println("Training an NN with GenAlg and datasetPercentage=" + datasetPercentage);
          NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(currSet, network, measure);
          OptimizationAlgorithm oa = new StandardGeneticAlgorithm(200, 50, 50, nnop);
          FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
          double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0, predicted, actual;
          fit.train();
          end = System.nanoTime();
          trainingTime = end - start;
          trainingTime /= Math.pow(10,9);
          Instance optimalInstance = oa.getOptimal();
          network.setWeights(optimalInstance.getData());

          start = System.nanoTime();
          for(int j = 0; j < test_instances.length; j++) {
            network.setInputValues(test_instances[j].getData());
            network.run();
            predicted = Double.parseDouble(test_instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
          }
          end = System.nanoTime();
          testingTime = end - start;
          testingTime /= Math.pow(10,9);
          testErr[curr] = 100 - (correct/(correct+incorrect)*100);
          System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
          testTime[curr] = testingTime;
      }
      System.out.println("--------GENETIC ALGS:------");
      System.out.println("Test Errors");
      for(double y : testErr) System.out.println(y + ",");
      System.out.println("Train Time");
      for(double x : trainTime) System.out.println(x + ",");
      System.out.println("Test Time");
      for(double x : testTime) System.out.println(x + ",");
      curr = 0;
      testErr = new double[6];
      testTime = new double[6];
      trainTime = new double[6];
      for (double datasetPercentage : datasetPercentages) {
          // Set up dataset.
          int lastIndex = (int)(train_instances.length * datasetPercentage);
          Instance[] curr_train_instances = Arrays.copyOfRange(train_instances, 0, lastIndex);
          DataSet currSet = new DataSet(curr_train_instances);
          System.out.println("Training an NN with GenAlg and datasetPercentage=" + datasetPercentage);
          NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(currSet, network, measure);
          OptimizationAlgorithm oa = new SimulatedAnnealing(1E9, 0.3, nnop);
          FixedIterationTrainer fit = new FixedIterationTrainer(oa, 1500);
          double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
          fit.train();
          end = System.nanoTime();
          trainingTime = end - start;
          trainingTime /= Math.pow(10,9);
          Instance optimalInstance = oa.getOptimal();
          network.setWeights(optimalInstance.getData());

          double predicted, actual;
          start = System.nanoTime();
          for(int j = 0; j < test_instances.length; j++) {
            network.setInputValues(test_instances[j].getData());
            network.run();
            predicted = Double.parseDouble(test_instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
          }
          end = System.nanoTime();
          testingTime = end - start;
          testingTime /= Math.pow(10,9);
          testErr[curr] = 100 - (correct/(correct+incorrect)*100);
          System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
          testTime[curr] = testingTime;
      }
      System.out.println("Test Errors");
      for(double y : testErr) System.out.println(y + ",");
      System.out.println("Train Time");
      for(double x : trainTime) System.out.println(x + ",");
      System.out.println("Test Time");
      for(double x : testTime) System.out.println(x + ",");
      curr = 0;
      testErr = new double[6];
      testTime = new double[6];
      trainTime = new double[6];
      for (double datasetPercentage : datasetPercentages) {
          // Set up dataset.
          int lastIndex = (int)(train_instances.length * datasetPercentage);
          Instance[] curr_train_instances = Arrays.copyOfRange(train_instances, 0, lastIndex);
          DataSet currSet = new DataSet(curr_train_instances);
          System.out.println("Training an NN with GenAlg and datasetPercentage=" + datasetPercentage);
          NeuralNetworkOptimizationProblem nnop = new NeuralNetworkOptimizationProblem(currSet, network, measure);
          OptimizationAlgorithm oa = new RandomizedHillClimbing(nnop);
          FixedIterationTrainer fit = new FixedIterationTrainer(oa, 50);
          double start = System.nanoTime();
          fit.train();
          double end = System.nanoTime();
          double trainingTime = end - start;
          trainingTime /= Math.pow(10,9);
          Instance optimalInstance = oa.getOptimal();
          network.setWeights(optimalInstance.getData());

          double correct = 0;
          double incorrect = 0;
          double predicted, actual, testingTime;
          start = System.nanoTime();
          for(int j = 0; j < test_instances.length; j++) {
            network.setInputValues(test_instances[j].getData());
            network.run();
            predicted = Double.parseDouble(test_instances[j].getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());
            double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
          }
          end = System.nanoTime();
          testingTime = end - start;
          testingTime /= Math.pow(10,9);
          testErr[curr] = 100 - (correct/(correct+incorrect)*100);
          System.out.println("Test Error: " + df.format(100 - (correct/(correct+incorrect)*100)));
          testTime[curr] = testingTime;
      }
      System.out.println("Test Errors");
      for(double y : testErr) System.out.println(y + ",");
      System.out.println("Train Time");
      for(double x : trainTime) System.out.println(x + ",");
      System.out.println("Test Time");
      for(double x : testTime) System.out.println(x + ",");
    }


  private static double train(OptimizationAlgorithm oa, BackPropagationNetwork network, Instance[] testing, int trainingIterations) {
    double length = testing.length;
    double incorrect = 0;
    for(int i = 0; i < trainingIterations; i++) {
      incorrect = 0;
      oa.train();
      double error = 0;
      for(int j = 0; j < testing.length; j++) {
        network.setInputValues(testing[j].getData());
        network.run();
        Instance output = testing[j].getLabel(), example = new Instance(network.getOutputValues());
        example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
        error += measure.value(output, example);
        incorrect += Math.abs(Double.parseDouble(output.toString()) - Double.parseDouble(example.getLabel().toString())) < 0.5 ? 0 : 1;
      }
    }
    return (incorrect/length*100);
  }

  private static Instance[] initializeInstances(Boolean test) {
    double[][][] attributes = new double[7738][][];
    String name = "datasetTrainNorm.csv";
    if (test) {
      System.out.println("here");
      attributes = new double[3317][][];
      name = "datasetTestNorm.csv";
    }
    try {
      BufferedReader br = new BufferedReader(new FileReader(new File(name)));
      // adding now
      // System.out.println(br.readLine());
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
