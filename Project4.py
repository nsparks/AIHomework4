from random import random
from random import shuffle
import matplotlib.pyplot as plt

global MUTATION_RATE

GENOME_LENGTH = 200
MAX_ITERATIONS = 20000
POPULATION_SIZE = 10
MUTATION_SIZE = 5
BATCH_PRINT = int(MAX_ITERATIONS / 10.0)
MUTATION_SEARCH_WIDTH = 10
MUTATION_MAX = 1.0
#MUTATION_RATE = 0.3

def fitness(chromosome):
	return float(sum(chromosome)) / float(GENOME_LENGTH)

def blend(parentOne, parentTwo):
	retval = []
	for gene in xrange(GENOME_LENGTH):
		if random() < 0.5:
			retval += [parentOne[gene]]
		else:
			retval += [parentTwo[gene]]
	return retval

def generateChromosome():
	chromosome = [0 if random() >= 0.5 else 1 for x in xrange(GENOME_LENGTH)]
	return chromosome

def randomGene():
	return 1 if random() >= 0.5 else 0

def crossoverMutate(best, genome, MUTATION_RATE):
	return blend(best, genome)

def generationMutate(best, genome, MUTATION_RATE):
	return generateChromosome() if random() < MUTATION_RATE else genome

def bigMutate(best, genome, MUTATION_RATE):
	return [1 - x if random() < MUTATION_RATE else x for x in genome]

def littleMutate(best, genome, MUTATION_RATE):
	return [1 - x if random() < MUTATION_RATE else x for x in best]

def selectionMutate(best, genome, MUTATION_RATE):
	retval = []
	for gene in xrange(len(best)):
		if random() < MUTATION_RATE:
			retval += [genome[gene]]
		else:
			retval += [best[gene]]
	return retval

def crossover(populationSlice):
	"""Note: gives back one smaller of a population"""
	retval = []
	for idx in xrange(len(populationSlice) - 1):
		first, second = populationSlice[idx], populationSlice[idx + 1]
		retval += [blend(first, second)]
	return retval

def populationFitness(population):
	fitnesses = map(fitness, population)
	averageFitness = float(sum(fitnesses)) / float(len(fitnesses))
	return averageFitness

def printStatus(population, iteration):
	averageFitness = populationFitness(population)
	print 'Averge fitness at iteration ', iteration, ': ', averageFitness

population = [generateChromosome() for x in xrange(POPULATION_SIZE)]
theMutate = crossoverMutate

for mutationMultiplier in range(MUTATION_SEARCH_WIDTH):
	MUTATION_RATE = (float(mutationMultiplier) / float(MUTATION_SEARCH_WIDTH)) * MUTATION_MAX

	fitnessTimeline = []
	print 'Mutation rate : ', MUTATION_RATE
	for generation in xrange(MAX_ITERATIONS):
		if generation % BATCH_PRINT == 0:
			printStatus(population, generation)
			fitnessTimeline += [populationFitness(population)]
		fitnesses = map(fitness, population)
		zipped = zip(population, fitnesses)
		sortedPopulation = sorted(zipped, key=lambda x: x[1])[::-1]

		selected = (zip(*sortedPopulation)[0])[0]

		toBeMutated = zip(*sortedPopulation[MUTATION_SIZE:])[0]
		mutated = [theMutate(selected, x, MUTATION_RATE) for x in toBeMutated]

		crossed = crossover(zip(*sortedPopulation[:MUTATION_SIZE])[0])
		
		population = [selected] + crossed + mutated
		shuffle(population)

	fitnesses = map(fitness, population)
	zipped = zip(population, fitnesses)
	sortedPopulation = sorted(zipped, key=lambda x: x[1])[::-1]
	finalFitness = sortedPopulation[0][1]
	plt.plot(range(len(fitnessTimeline)), fitnessTimeline, label="Fitness : {0} (Mutation rate {1})".format(finalFitness, MUTATION_RATE))
	
plt.axis([0, len(fitnessTimeline), 0, 1])
plt.legend()
plt.show()
