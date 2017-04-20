package me.cassioalmeidas.ga;

import java.util.Random;

public class Individual implements Comparable<Individual>{
	
	public double[] chromossome;
	public double error;
	
	private int numGenes;
	private double minGene;
	private double maxGene;
	private double mutateRate;
	private double mutateChange;
	
	static Random rnd = new Random();

	public Individual(int numGenes, double minGene, double maxGene, double mutateRate, 
			double mutateChange){
		this.numGenes = numGenes;
		this.minGene = minGene;
		this.maxGene = maxGene;
		this.mutateRate = mutateChange;
		this.mutateChange = mutateChange;
		this.chromossome = new double[numGenes];
		
		for(int i = 0; i < this.chromossome.length; ++i)
			this.chromossome[i] = (maxGene - minGene) * rnd.nextDouble() + minGene;
		
	}
	
	@Override
	public int compareTo(Individual other) {
		// TODO Auto-generated method stub
		if(this.error < other.error) return -1;
		else if(this.error > other.error) return 1;
		else return 0;
	}
	
	public double getError() {
		return error;
	}

	public void setError(double error) {
		this.error = error;
	}

	public int getNumGenes() {
		return numGenes;
	}

	public void setNumGenes(int numGenes) {
		this.numGenes = numGenes;
	}

	public double getMinGene() {
		return minGene;
	}

	public void setMinGene(double minGene) {
		this.minGene = minGene;
	}

	public double getMaxGene() {
		return maxGene;
	}

	public void setMaxGene(double maxGene) {
		this.maxGene = maxGene;
	}

	public double getMutateRate() {
		return mutateRate;
	}

	public void setMutateRate(double mutateRate) {
		this.mutateRate = mutateRate;
	}

	public double getMutateChange() {
		return mutateChange;
	}

	public void setMutateChange(double mutateChange) {
		this.mutateChange = mutateChange;
	}
	
}
