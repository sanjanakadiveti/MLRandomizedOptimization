import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
// import FixedIterationTrainerMod;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double start, trainingTime, end = 0;

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainerMod fit = new FixedIterationTrainerMod(rhc, 200000);
        start = System.nanoTime();
        fit.train(start);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + " Time: " + trainingTime);

        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainerMod(sa, 200000);
        start = System.nanoTime();
        fit.train(start);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        System.out.println("SA: " + ef.value(sa.getOptimal()) + " Time: " + trainingTime);

        double[] vals = new double[10];
        int curr = 0;
        for (double i = 0.1; i < 1; i += 0.1) {
          SimulatedAnnealing sa = new SimulatedAnnealing(1E11, i, hcp);
          FixedIterationTrainerMod fit = new FixedIterationTrainerMod(sa, 200000);
          start = System.nanoTime();
          fit.train(start);
          end = System.nanoTime();
          trainingTime = end - start;
          trainingTime = end - start;
          trainingTime /= Math.pow(10,9);

          System.out.println("SA: " + ef.value(sa.getOptimal()) + " Time: " + trainingTime);
          vals[curr] =  ef.value(sa.getOptimal());
          curr += 1;
        }
        for (double d : vals) System.out.print(d + ",");

        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainerMod(ga, 200000);
        start = System.nanoTime();
        fit.train(start);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);


        System.out.println("GA: " + ef.value(ga.getOptimal()) + " Time: " + trainingTime);

        MIMIC mimic = new MIMIC(200, 20, pop);
        fit = new FixedIterationTrainerMod(mimic, 20000);
        start = System.nanoTime();
        fit.train(start);
        end = System.nanoTime();
        trainingTime = end - start;
        trainingTime = end - start;
        trainingTime /= Math.pow(10,9);

        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + " Time: " + trainingTime);
    }
}
