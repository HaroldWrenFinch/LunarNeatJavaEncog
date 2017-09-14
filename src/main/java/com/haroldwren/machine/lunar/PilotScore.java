package com.haroldwren.machine.lunar;

import org.encog.ml.CalculateScore;
import org.encog.ml.MLMethod;
import org.encog.neural.neat.NEATNetwork;

public class PilotScore implements CalculateScore {

    @Override
    public double calculateScore(MLMethod network) {
        NeuralPilot pilot = new NeuralPilot((NEATNetwork) network, false);
        return pilot.scorePilot();
    }


    public boolean shouldMinimize() {
        return false;
    }


    @Override
    public boolean requireSingleThreaded() {
        return false;
    }
}
