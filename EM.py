import numpy as np

# TODO:
	# Make lambda be initialised properly (instead of all 0s)
	# Get it to loop (currently it only does one iteration), need to start thinking about
	# Try researching Baum-Welch intialisation
	# If we start from all 1s then we end up with something pretty good (as in probability > 0 for L = 1000)
		# Consider this as an option (could go into some detail about why it was chosen in the report)
			# i.e. traditionally, Baum-Welch begins from a random set of parameters but in this case, with larger L, the initial probability is too small to even be represented in floating point (so it is truncated to 0). Beginning from 0 probability, the algorithm cannot make any improvements
			# Although all 1s is not a valid Markov model, it does give the algorithm a starting point from which it is able to build a valid model which produces the sequence with high probability
			# 10^-300 may sound miniscule but considering that the distribution used to generate X in this case is uniform and (1/4)^1000 ~ 1.14 * 10^-600 it's actually really good
		# For X = all 1s this gives us probability of 0.991 even for L = 1000
	# Test it
		# Try testing it by building a markov model (randomly) then generating a sequence using that.
			# Probably generating them randomly is what's messing it up (consider probabilities of specific uniformly random sequences)
	# Bugs:
		# gamma[l] sometimems sums to 0 (before normalisation) (I've patched this by setting gamma[l] to 1 in this case)
		# same for xi[l]

def viterbis():
	"""Returns the probability of a most probable path that produces X"""
	v = np.zeros((k, L))
	for i in range(1, k):
		v[i][0] = 0
	v[0][0] = 1
	for i in range(1, L):
		for l in range(k):
			v[l][i] = e[l][X[i]] * np.array([v[s][i - 1] * m[s][l] for s in range(k)]).max()
	return np.array([v[l][L - 1] for l in range(k)]).max()

k = 10
L = 1000
alphabet_size = 4

X = np.random.randint(0, alphabet_size / 2, (L), dtype=int)
#X = np.zeros((L), dtype=int)

pi = np.ones((k))
m = np.ones((k,k))
e = np.ones((k,alphabet_size))

'''pi = np.full((k), 1/k)
m = np.full((k,k), 1/k)
e = np.full((k,alphabet_size), 1/alphabet_size)'''

'''pi = np.random.uniform(0, 1, (k))
pi_s = pi.sum()
for i in range(k):
	pi[i] /= pi_s

m = np.random.uniform(0, 1, (k,k))
for i in range(k):
	s = m[i].sum()
	for j in range(k):
		m[i][j] /= s

e = np.random.uniform(0, 1, (k,alphabet_size))
for i in range(k):
	s = e[i].sum()
	for j in range(alphabet_size):
		e[i][j] /= s'''

print(viterbis())

for x in range(25):
	print(x)
	alpha = np.zeros((L,k))
	beta = np.zeros((L,k))
	gamma = np.zeros((L,k))
	xi = np.zeros((L,k,k))

	#gamma_prime = np.zeros((L))
	xi_prime = np.zeros((L))

	for i in range(k):
		alpha[0][i] = pi[i] * e[i][X[0]]
		beta[L - 1][i] = 1

	for i in range(k):
		for l in range(1,L):
			s = 0
			for j in range(k):
				s += alpha[l - 1][j] * m[j][i]
			alpha[l][i] = e[i][X[l]] * s
		for l in range(L - 2, -1, -1):
			s = 0
			for j in range(k):
				s += beta[l+1][j] * m[i][j] * e[j][X[l+1]]
			beta[l][i] = s

	gamma = alpha * beta

	for l in range(L):
		s = gamma[l].sum()
		if s == 0:
			gamma[l] = np.ones((k))
		else:
			gamma[l] /= gamma[l].sum()

	for i in range(k):
		for j in range(k):
			for l in range(L - 1):
				xi[l][i][j] = alpha[l][i] * m[i][j] * e[j][X[l + 1]] * beta[l + 1][j]
				xi_prime[l] += xi[l][i][j]

	for l in range(L - 1):
		if xi_prime[l] == 0:
			xi[l] = np.ones((k, k))
		else:
			xi[l] /= xi_prime[l]

	for i in range(k):
		pi[i] = gamma[0][i]
		s = 0
		for l in range(L - 1):
			s += gamma[l][i]
		if s != 0:  # Why is s sometimes 0?
			for j in range(k):
				s2 = 0
				for l in range(L - 1):
					s2 += xi[l][i][j]
				m[i][j] = s2 / s
			s += gamma[L - 1][i]
			for b in range(alphabet_size):
				s2 = 0
				for l in range(L):
					if X[l] == b:
						s2 += gamma[l][i]
				e[i][b] = s2 / s
	for i in range(k):
		m[i] /= m[i].sum()

print(viterbis())