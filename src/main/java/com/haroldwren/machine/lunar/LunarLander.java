package com.haroldwren.machine.lunar;

import org.encog.Encog;
import org.encog.ml.CalculateScore;
import org.encog.ml.ea.train.EvolutionaryAlgorithm;
import org.encog.neural.neat.NEATNetwork;
import org.encog.neural.neat.NEATPopulation;
import org.encog.neural.neat.NEATUtil;

public class LunarLander {

    public static void main(String args[]) {
        NEATPopulation pop = new NEATPopulation(3,1,500);
        pop.setInitialConnectionDensity(1.0); // not required, but speeds training
        pop.reset();

        CalculateScore score = new PilotScore();
        final EvolutionaryAlgorithm train = NEATUtil.constructNEATTrainer(pop,score);

        double error;

        do {
            train.iteration();
            int iteration = train.getIteration();
            error = train.getError();
            System.out.println("Epoch #" + iteration + " Score:" + error + ", Species:" + pop.getSpecies().size());
        } while(error < 8200);

        NEATNetwork network = (NEATNetwork)train.getCODEC().decode(train.getBestGenome());

        System.out.println("\nHow the winning network landed:");
        NeuralPilot pilot = new NeuralPilot(network,true);
        System.out.println(pilot.scorePilot());
        Encog.getInstance().shutdown();
    }
}
