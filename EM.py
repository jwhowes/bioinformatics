import numpy as np

# This whole thing doesn't work

# TODO:
	# Something is wrong with my implementation of Viterbi's
		# But I'm fairly sure (logically) that alpha[L - 1].sum() is the value I'm looking for anyway (ofc at the end of an iteration)
	# Try researching Baum-Welch intialisation
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

k = 15
L = 100
alphabet_size = 10

epsilon = 1.00001

#X = np.random.randint(0, alphabet_size, (L), dtype=int)
X = np.array([i % alphabet_size for i in range(L)])

#pi = np.ones((k))
#m = np.ones((k,k))
#e = np.ones((k,alphabet_size))

'''pi = np.array([1, 0, 0, 0], dtype=float)

m = np.array([	[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1],
				[1, 0, 0, 0]], dtype=float)

e = np.array([	[1, 0, 0, 0],
				[0, 1, 0, 0],
				[0, 0, 1, 0],
				[0, 0, 0, 1]], dtype=float)'''

'''pi = np.full((k), 1/k)
m = np.full((k,k), 1/k)
e = np.full((k,alphabet_size), 1/alphabet_size)'''

pi = np.random.uniform(0, 1, (k))
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
		e[i][j] /= s
t = 0
p = 0
while True:
	t += 1
	print(t, p)
	alpha = np.zeros((L,k))
	beta = np.zeros((L,k))
	gamma = np.zeros((L,k))
	xi = np.zeros((L,k,k))

	#gamma_prime = np.zeros((L))
	xi_prime = np.zeros((L))

	for i in range(k):
		alpha[0][i] = pi[i] * e[i][X[0]]
		beta[L - 1][i] = 1
	for l in range(1, L):
		for i in range(k):
			s = 0
			for j in range(k):
				s += alpha[l - 1][j] * m[j][i]
			alpha[l][i] = e[i][X[l]] * s
	for l in range(L - 2, -1, -1):
		for i in range(k):
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
	new_p = alpha[L - 1].sum()
	if p > 0 and new_p / p <= epsilon:
		break
	p = new_p

print(alpha[L - 1].sum())