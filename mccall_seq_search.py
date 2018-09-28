'''
Mccall Job Search


Algorithm solving the bellman equation for a
job seeker


bellman:
v_i = max{(w_i/(1 - beta)), (c + beta*sum(v_i * p_i))}

Algorithm:
i) Pick an arbitrary v_i
ii) Solve the bellman's rhs
iii) check  if |v_i' -  v_1| < thresh
iv)if not return to step 2 after setting v_i' = v_i

w_i and beta are given to us. w_i is drawn randomly from
a uniform distribution, and beta is constant at set arbitratily
at 0.99
'''

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class mccall():

	def __init__(self, 
			w = np.random.uniform(high = 100, 
								low=1, 
								size = 100),
			beta = 0.99):
		'''
		set initial values of parameters in argument
		function returns wage pdf
		'''
		w.sort()
		wmean = np.mean(w)
		wstd = np.std(w)
		pdf = stats.norm.pdf(w, wmean, wstd)
		self.w, self.beta, self.pdf = w, beta, pdf

	def plot_wages(self):
		fig, ax = plt.subplots()
		
		ax.plot(self.w, self.pdf,
			alpha = 0.5)
		ax.set_xlabel('Wages')
		ax.set_ylabel('Probabilties')
		plt.show()

	def calc(self, n_iter = 100, tol = 1e-6, c = 25):
		w, beta, pdf= self.w, self.beta, self.pdf
		v = w/(1 - beta)
		v_dash = np.empty_like(v)
		error = 100
		i = 0
		while i < n_iter and error > tol:
			for j, w_val in enumerate(w):
				v_dash[j] = max((w_val/(1 - beta)), np.sum(v * pdf))
	
			error = np.max(np.absolute(v_dash - v))
			v = v_dash
			i += 1

		r = (1 - beta) * (c + beta * np.sum(v * pdf))

		return print(f"Reservation Wage: {r}")


if __name__ == '__main__':
	mdl = mccall()
	mdl.calc()