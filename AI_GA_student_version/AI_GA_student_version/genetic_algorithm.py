from gameplay import * 
import player
import numpy as np
from random import randint
import random 
import copy


class GeneticAlgorithm:

	def __init__(self, population_size = 100, probability_game = False, use_multinomial = False):
		self.probability_game = probability_game
		self.use_multinomial = use_multinomial
		self.population_size = population_size
		self.population = [Player("Student"+str(i)) for i in range(0,population_size)]
		self.game = GamePlay()

	def run_genetic_algorithm(self, generations = 10):

		for i in range(0, generations):
			self.compute_population_fitness()
			average_fitness = 0
			for j in self.population:
				average_fitness += j.fitness_value
			print("====Generation " + str(i) + " average fitness: " + str(average_fitness/len(self.population))+"====")
			if (i < generations-1):
				self.reproduce()

		self.population.sort(key=lambda x:x.fitness_value, reverse=True)
		#take the fittest
		fittest_player = self.population[0]
		print("Fittest player fitness value: " + str(fittest_player.fitness_value))
		return fittest_player


	def reproduce(self):

		###TO DO###
		## This method creates a new generation ##
		## 1. Get the top x% fittest players 
		## 2. Create a new population based on these "top-fit" parents
		## 3. The new population can be comprised of A% of clone, B% of mutation, and C% of 
		##    crossovers of these "top-fit parents"
		self.population.sort(key=lambda x:x.fitness_value, reverse=True)
		parent_portion = int(0.1 * len(self.population)) # 10%
		topfit_parents = self.population[:parent_portion]

		topfit_normalizer = sum(parent.fitness_value for parent in topfit_parents)
		topfit_probability = [parent.fitness_value / topfit_normalizer for parent in topfit_parents]
		
		clone_limit = int(0.2 * len(self.population)) # 10%
		mutate_limit = clone_limit + int(0.2 * len(self.population)) # the next 10%

		new_population = []

		for i in range(len(self.population)):
			# parent = np.random.choice(topfit_normalizer, topfit_probability)
			if i < clone_limit:
				new_population.append(self.clone(topfit_parents))
			elif i < mutate_limit:
				new_population.append(self.mutation(topfit_parents))
			else:
				new_population.append(self.crossover(topfit_parents))

		self.population = new_population

		return None

	def compute_population_fitness(self, days=100):
		index = 0
		for player in self.population:
			self.population[index].fitness_value = self.get_average_survival_time(player)
			# if (index % 10 == 0):
			# 	print(str(index)+" done")
			index += 1

	def get_average_survival_time(self, player, days = 100, repeat = 5):
		total_days_survived = 0
		for i in range(0, repeat):
			total_days_survived	+= self.game.play(player, days, probability_game = self.probability_game, use_multinomial = self.use_multinomial, printout = False)
		return total_days_survived/repeat
		
	## child is an exact copy of the parent	
	def clone(self, parent_list):
		parent = parent_list[random.randrange(1, len(parent_list))]
		child = copy.deepcopy(parent)
		return child

	## child is a mutation of a single parent
	def mutation(self, parent_list):
		parent = parent_list[random.randrange(1, len(parent_list))]
		child = copy.deepcopy(parent)
		child.mutate()
		return child

	## child is a product of crossover between two parents
	def crossover(self, parent_list):
		father = parent_list[random.randrange(1,len(parent_list))]
		mother = parent_list[random.randrange(1,len(parent_list))]
		mother.marry(father)
		child = copy.deepcopy(mother)
		return child


if __name__ == '__main__':
	probability_game = True
	use_multinomial= True

	G = GeneticAlgorithm(population_size = 100, probability_game = probability_game, use_multinomial = use_multinomial)
	fittest_player = G.run_genetic_algorithm(10)
	fittest_player.attributes.print_attributes()

	## try out the performance of the fittest player several times
	G.game.play(fittest_player, 50, probability_game = probability_game, use_multinomial = use_multinomial, printout = True)
	G.game.play(fittest_player, 50, probability_game = probability_game, use_multinomial = use_multinomial, printout = True)
	G.game.play(fittest_player, 50, probability_game = probability_game, use_multinomial = use_multinomial, printout = True)
	G.game.play(fittest_player, 50, probability_game = probability_game, use_multinomial = use_multinomial, printout = True)


	




