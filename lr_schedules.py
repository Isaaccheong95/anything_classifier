'''
Implements different learning rate schedules.
'''
import numpy as np

def cyclical_lr(base_lr, max_lr, stepsize, decrease_lr_by=0.10):
	# This particular implementation will decrease alternate the lr and momentum of the Adam optimizer, i.e. when lr increases, momentum decreases, etc.
	# stepsize: Number of iterations per half cycle, where 'cycle' refers to 1 cycle in this cyclical learning rate schedule.

	while True:
		# lr_factor and mom_factor define the amounts by which we will change the lr and momentum.
		lr_factor = (max_lr - base_lr) / stepsize # we execute this step in the loop because we need to change the lr_factor during the annealing phase
		# at the end of each cycle, where the lr will decrease at a slower rate.

		if base_lr > max_lr:
			raise ValueError("max_lr must be more than base_Lr.")

		curr_lr = base_lr

		# first half of the cycle: lr increases, momentum decreases
		for i in range(stepsize):
			curr_lr += lr_factor
			yield curr_lr # "yield" will return a generator which will iterate through its values only onc as it does not store them in memory.
			# This code does not un when you call the cyclical_lr() function, but it returns a generator object. When you iterate through that generator object
			# only then will this code run and return the lr and mom values.

		# second half of the cycle: lr decreases, momentum increases
		for i in range(stepsize):
			curr_lr -= lr_factor
			yield curr_lr

		# annealing phase: we decrease the lr by "decrease_lr_by" amount, which is a percentage of the original lr, over a period of a number of iterations which is "half_stepsize".
		# We then set this new decreased base lr to be the new lr on the next cycle.
		# During this annealing phase, momentum remains constant.
		half_stepsize = stepsize //2
		new_base_lr = base_lr - (base_lr * decrease_lr_by)
		new_lr_factor = (base_lr - new_base_lr) / half_stepsize

		for i in range(half_stepsize):
			curr_lr -= new_lr_factor
			yield curr_lr

		base_lr = new_base_lr

def exp_decay(base_lr, curr_epoch, k=0.2):
	# Reduces the base learning rate exponentially every epoch.
	return base_lr * np.exp(-k * curr_epoch)

def step_decay(base_lr, curr_epoch, drop_factor=0.5, drop_epoch=10):
	# Reduces the learning rate by "drop_factor" every "drop_epoch".
	return base_lr * (drop_factor ^ (curr_epoch // drop_epoch))
