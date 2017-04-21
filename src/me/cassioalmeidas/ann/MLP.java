package me.cassioalmeidas.ann;

import java.util.Arrays;
import java.util.Random;

import me.cassioalmeidas.ga.Individual;

public class MLP {
	private int numInput;
	private int numHidden;
	private int numOutput;
	
	private double[] inputs;
	private double[][] ihWeights;
	private double[] hBiases;
	private double[] hOutputs;
	private double[][] hoWeights;
	private double[] oBiases;
	private double[] outputs;
	
	private Random rnd;
	
	
	public MLP(int numInput, int numHidden, int numOutput){
		this.numInput = numInput;
		this.numHidden = numHidden;
		this.numOutput = numOutput;
		this.inputs = new double[numInput];
		this.ihWeights = makeMatrix(numInput, numHidden);
		this.hBiases = new double[numHidden];
		this.hOutputs = new double[numHidden];
		this.hoWeights = makeMatrix(numHidden, numOutput);
		this.oBiases = new double[numOutput];
		this.outputs = new double[numOutput];
		this.rnd = new Random();
	}
	
	public static double[][] makeMatrix(int rows, int cols){
		double[][] result = new double[rows][];
		for(int i = 0; i < result.length; i++)
			result[i] = new double[cols];
		return result;
	}
	
	public void setWeights(double[] weights) throws Exception{
		int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
		
		if(weights.length != numWeights)
			throw new Exception("Bad weight array length!");
		
		int k = 0;
		
		for(int i = 0; i < numInput; ++i)
			for(int j = 0; j < numHidden; ++j)
				ihWeights[i][j] = weights[k++];
		for(int i = 0; i < numHidden; ++i)
			hBiases[i] = weights[k++];
		for(int i = 0; i < numHidden; ++i)
			for(int j = 0; j < numOutput; ++j)
				hoWeights[i][j] = weights[k++];
		for(int i = 0; i < numOutput; ++i)
			oBiases[i] = weights[k++];
	}
	
	public double[] getWeights(){
		int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
		double[] result = new double[numWeights];
		int k = 0;
		
		for (int i = 0; i < ihWeights.length; ++i)
			for (int j = 0; j < ihWeights[0].length; ++j)
	          result[k++] = ihWeights[i][j];
		for (int i = 0; i < hBiases.length; ++i)
			result[k++] = hBiases[i];
	    for (int i = 0; i < hoWeights.length; ++i)
	        for (int j = 0; j < hoWeights[0].length; ++j)
	          result[k++] = hoWeights[i][j];
	    for (int i = 0; i < oBiases.length; ++i)
	    	result[k++] = oBiases[i];
	    return result;
	}
	
	public double[] computeOutputs(double[] xValues) throws Exception{
		// feed-forward mechanism for NN classifier
	      if (xValues.length != numInput)
	        throw new Exception("Bad xValues array length");

	      double[] hSums = new double[numHidden];
	      double[] oSums = new double[numOutput];

	      for (int i = 0; i < xValues.length; ++i)
	        this.inputs[i] = xValues[i];

	      for (int j = 0; j < numHidden; ++j)
	        for (int i = 0; i < numInput; ++i)
	          hSums[j] += this.inputs[i] * this.ihWeights[i][j];

	      for (int i = 0; i < numHidden; ++i)
	        hSums[i] += this.hBiases[i];

	      for (int i = 0; i < numHidden; ++i)
	        this.hOutputs[i] = hyperTan(hSums[i]);
	      
	      for (int j = 0; j < numOutput; ++j)
	        for (int i = 0; i < numHidden; ++i)
	          oSums[j] += hOutputs[i] * hoWeights[i][j];


	      for (int i = 0; i < numOutput; ++i)
	    	  oSums[i] += oBiases[i];
	      
	      if(numOutput > 1){
	    	  double[] softOut = softmax(oSums);
		      System.arraycopy(softOut, 0, this.outputs, 0, softOut.length);
	      }else{
	    	  System.arraycopy(oSums, 0, this.outputs, 0, oSums.length);
	      }
	      
	      
	      double[] retResult = new double[numOutput];

	      System.arraycopy(this.outputs, 0, retResult, 0, retResult.length);
	      
	      return retResult;
	}

	private double hyperTan(double x) {
		if (x < -20.0) return -1.0;
		else if (x > 20.0) return 1.0;
		else return Math.tanh(x);
	}

	private double[] softmax(double[] oSums) {
		double max = oSums[0];
		
		for(int i = 0; i < oSums.length; ++i)
			if (oSums[i] > max) max = oSums[i];
		
		double scale = 0.0;
		for(int i = 0; i < oSums.length; ++i)
			scale += Math.exp(oSums[i] - max);
		
		double[] result = new double[oSums.length];
		for(int i = 0; i < oSums.length; ++i)
			result[i] = Math.exp(oSums[i]-max) / scale;
		
		return result;
	}
	
	public double[] evolve(double[][] trainData, int popSize, int maxGeneration,
			double exitError,double mutateRate, double mutateChange, double crossoverRate, double tau) throws Exception{
		int numWeights = (this.numInput * this.numHidden) + (this.numHidden * this.numOutput) + this.numHidden + this.numOutput;
		double minX = -10.0;
		double maxX = 10.0;
		
		Individual[] population = new Individual[popSize];
		double[] bestSolution = new double[numWeights];
		double bestError = Double.MAX_VALUE;
		
		for(int i = 0; i < population.length; ++i){
			population[i] = new Individual(numWeights, minX, maxX, mutateRate, mutateChange);
			population[i].error = meanSquaredError(trainData, population[i].chromossome);
			if (population[i].error < bestError){
				bestError = population[i].error;
				System.arraycopy(population[i].chromossome, 0, bestSolution, 0, numWeights);
			}
		}
		
		int gen = 0;
		boolean done = false;
		
		while(gen < maxGeneration && done == false){
			Individual[] parents = select(2, population, tau);
			Individual[] babies = new Individual[2];
			if (rnd.nextDouble() < crossoverRate){
				babies = reproduce(parents[0], parents[1], minX, maxX,mutateRate, mutateChange);
			}else{
				babies[0] = parents[0];
				babies[1] = parents[1];
			}
			babies[0].error = meanSquaredError(trainData, babies[0].chromossome);
			babies[1].error = meanSquaredError(trainData, babies[1].chromossome);
			
			place(babies[0], babies[1], population);
			
			Individual immigrant = new Individual(numWeights, minX, maxX, mutateRate, 
					mutateChange);
			immigrant.error = meanSquaredError(trainData, immigrant.chromossome);
			population[population.length-3] = immigrant;
			
			for(int i = popSize-3; i < popSize; ++i){
				if(population[i].error < bestError){
					bestError = population[i].error;
					System.arraycopy(population[i].chromossome, 0, bestSolution, 0, population[i].chromossome.length);
					if(bestError < exitError){
						done = true;
						System.out.println("Early exit at generation: "+gen);
					}
				}
			}
			++gen;
			if(gen % 1000000 == 0 || gen <= 1000000)
				System.out.println(gen+","+bestError);
		}
		
		return bestSolution;
	}

	private static void place(Individual b1, Individual b2, Individual[] population) {
		int popSize = population.length;
		Arrays.sort(population);
		population[popSize-1] = b1;
		population[popSize-2] = b2;
	}

	private Individual[] reproduce(Individual p1, Individual p2, double minGene, double maxGene,
			double mutateRate, double mutateChange) {
		int numGenes = p1.chromossome.length; 
		int pivot = rnd.nextInt(numGenes);
		Individual baby1 = new Individual(numGenes, minGene, maxGene, mutateRate, mutateChange);
		Individual baby2 = new Individual(numGenes, minGene, maxGene, mutateRate, mutateChange);
		
		for(int i = 0; i <= pivot; ++i)
			baby1.chromossome[i] = p1.chromossome[i];
		for(int i = pivot + 1; i < numGenes; ++i)
			baby2.chromossome[i] = p1.chromossome[i];
		for(int i = 0; i <= pivot; ++i)
			baby2.chromossome[i] = p2.chromossome[i];
		for(int i = pivot + 1; i < numGenes; ++i)
			baby1.chromossome[i] = p2.chromossome[i];
		
		mutate(baby1, maxGene, mutateRate, mutateChange);
		mutate(baby2, maxGene, mutateRate, mutateChange);
		
		Individual[] result = new Individual[2];
		result[0] = baby1;
		result[1] = baby2;
		
		return result;
	}

	private void mutate(Individual baby, double maxGene, double mutateRate, double mutateChange) {
		double hi = mutateChange * maxGene;
		double lo = -hi;
		for(int i = 0; i < baby.chromossome.length; ++i){
			if(rnd.nextDouble() < mutateRate){
				double delta = (hi - lo) * rnd.nextDouble() + lo;
				baby.chromossome[i] += delta;
			}
		}
	}

	private Individual[] select(int n, Individual[] population, double tau) {
		int popSize = population.length;
		int[] indexes = new int[popSize];
		for(int i = 0; i < indexes.length; ++i)
			indexes[i] = i;
		for(int i = 0; i < indexes.length; ++i){ // shuffle
			int r = i + rnd.nextInt((indexes.length - i));
			int tmp = indexes[r];
			indexes[r] = indexes[i];
			indexes[i] = tmp;
		}
		
		int tournSize = (int)(tau * popSize);
		if(tournSize < n ) tournSize = n;
		Individual[] candidates = new Individual[tournSize];
		
		for(int i = 0; i < tournSize; ++i)
			candidates[i] = population[indexes[i]];
		Arrays.sort(candidates);

		Individual[] results = new Individual[n];
		
		for(int i = 0; i < n; ++i)
			results[i] = candidates[i];
		
		return results;
	}

	private double meanSquaredError(double[][] trainData, double[] weights) throws Exception {

		this.setWeights(weights);
		double[] xValues = new double[numInput];
		double[] tValues = new double[numOutput];
		double sumSquaredError = 0.0;
		
		for(int i = 0; i < trainData.length; ++i){
			System.arraycopy(trainData[i], 0, xValues, 0, numInput);
			System.arraycopy(trainData[i], numInput, tValues, 0, numOutput);
			double[] yValues = this.computeOutputs(xValues);
			for(int j = 0; j < yValues.length; ++j)
				sumSquaredError += Math.pow((yValues[j] - tValues[j]), 2);
		}
		return sumSquaredError/trainData.length;
	}
}
