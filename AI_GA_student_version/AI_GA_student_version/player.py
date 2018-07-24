from stats import *
import random
from random import randint

class Player:
	def __init__(self, name):
		self.id = name
		##["Strength", "Agility", "Intelligence", "Charm", "Vitality", "Stamina", "Spirit"]
		self.attributes = Attributes([0,0,0,0,0,0,0])
		self.randomize_attribute()
		self.fitness_value = 0

	#used to initialize attributes random;
	def randomize_attribute(self):
		maxAttribute = self.attributes.attribute_points_limit
		index = 0
		while (maxAttribute > 0) and (index < self.attributes.attribute_types):
			attribute_point = randint(0,3)
			assert(attribute_point < 4)
			if (maxAttribute - attribute_point >= 0) and (index < self.attributes.attribute_types):
				self.attributes.change_attribute(index, attribute_point)
				maxAttribute -= attribute_point
				index += 1

    #change one of the attributes 
	def mutate(self):

		## TODO ##
		## Implement a method to change any one of the attribute's skill point ##
		## Can be done randomly or deterministically ## 
		## Can also be done by swapping the skill point of one random attribute category to another ##
		
		# mutate here is:
	        # if sum(self.attributes.attribute_level) <= self.attributes.attribute_points_limit
	        # then one can increase one single attribute
	        # otherwise, one swaps the stats of two random attributes
	        # in the first case: mutate cannot increase an attribute to more than 4
	        # in the first case: mutate cannot decrease an attribute to less than zero

		if sum(self.attributes.get_all_attributes()) <= self.attributes.attribute_points_limit:
			possible_attr = [at for at in range(self.attributes.attribute_types)
							 if self.attributes.get_attribute(at) < 4]
			while True:
				attribute_to_increase = random.choice(possible_attr)
				increased_value = 1 + self.attributes.get_attribute(attribute_to_increase)
				if increased_value < 4: break
			
			self.attributes.change_attribute(attribute_to_increase, increased_value)
		else:
			# swap attributes
			attr_a = randint(0, self.attributes.attribute_types-1)
			attr_b = random.choice([at for at in range(self.attributes.attribute_types-1) if at != attr_a])
			val_a = self.attributes.get_attribute(attr_a)
			val_b = self.attributes.get_attribute(attr_b)

			self.attributes.change_attribute(attr_a, val_b)
			self.attributes.change_attribute(attr_b, val_a)
		
		return None

    #replace one's attribute with partner's
	def marry(self, significant_other):
		
		## TODO ##
		## Change half of the attributes with the attributes' of the significant_other ##
		
		# there are 7 attributes
		#important: after a swap of any attribute it must hold: 
		#sum(self.attributes.attribute_level) <= self.attributes.attribute_points_limit
		# if this does not hold, then undo the swap!        
		# stop swapping if 5 attributes were swapped, or if one has gone through all 7 attributes
		
		swapped = 0
		attr_indexes = list(range(self.attributes.attribute_types))
		random.shuffle(attr_indexes)

		for i in attr_indexes:
			my_val = self.attributes.get_attribute(i)
			so_val = significant_other.attributes.get_attribute(i)

			# swapping
			self.attributes.change_attribute(i, so_val)
			significant_other.attributes.change_attribute(i, my_val)

			# check conditions
			undo = False

			if sum(self.attributes.get_all_attributes()) > self.attributes.attribute_points_limit: 
				undo = True
			if sum(significant_other.attributes.get_all_attributes()) > significant_other.attributes.attribute_points_limit:
				undo = True

			if undo:
				self.attributes.change_attribute(i, my_val)
				significant_other.attributes.change_attribute(i, so_val)
			else:
				swapped += 1

			if swapped >= 5:
				break
		return None



			



		
