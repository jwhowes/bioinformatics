import numpy as np

# TODO:
	# Make lambda be initialised properly (instead of all 0s)
	# Test it

k = 10
L = 100
alphabet_size = 4

X = np.zeros((L), dtype=int)

pi = np.zeros((k))
m = np.zeros((k,k))
e = np.zeros((k,alphabet_size))

alpha = np.zeros((L,k))
beta = np.zeros((L,k))
gamma = np.zeros((L,k))
xi = np.zeros((L,k,k))

gamma_prime = np.zeros((L))
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
	gamma_prime[l] = gamma[l].sum()

for i in range(k):
	for j in range(k):
		for l in range(L - 1):
			xi[l][i][j] = alpha[l][i] * m[i][j] * e[j][X[l + 1]] * beta[l + 1][j]
			xi_prime[l] += alpha[l][i] * m[i][j] * e[j][X[l + 1]] * beta[l + 1][j]

for i in range(k):
	pi[i] = gamma[0][i]
	s = 0
	for l in range(L - 1):
		s += gamma[l][i]
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