import numpy as np

# This whole thing doesn't work

# TODO:
	# Try researching Baum-Welch intialisation
	# Test it
		# How to generate test sequences?

def forward():
	F = np.zeros((L, k))
	for i in range(k):
		F[0][i] = pi[i] * e[i][X[0]]
	#F[0][0] = 1
	for i in range(1, L):
		for l in range(k):
			s = 0
			for j in range(k):
				s += F[i - 1][j] * m[j][l]
			F[i][l] = e[l][X[i]] * s
	print(F)
	return F[L - 1].sum()

k = 4
L = 1000
alphabet_size = 2

X = np.random.randint(0, alphabet_size, (L), dtype=int)
#X = np.array([i % alphabet_size for i in range(L)])

X = np.zeros((L), dtype=int)

test_pi = np.random.uniform(0, 1, (k))
test_pi /= test_pi.sum()

test_m = np.random.uniform(0, 1, (k,k))
for i in range(k):
	test_m[i] /= test_m.sum()

test_e = np.random.uniform(0, 1, (k, alphabet_size))
for i in range(k):
	test_e[i] /= test_e[i].sum()

r = np.random.uniform(0, 1)
state = -1
for i in range(k):
	if r <= test_pi[i]:
		state = i
		break
	r -= test_pi[i]

for l in range(L):
	r = np.random.uniform(0, 1)
	for i in range(alphabet_size):
		if r <= test_e[state][i]:
			X[l] = i
			break
		r -= test_e[state][i]
	r = np.random.uniform(0, 1)
	for i in range(k):
		if r <= test_m[state][i]:
			state = i
			break
		r -= test_m[state][i]

#pi = np.ones((k))
#m = np.ones((k,k))
#e = np.ones((k,alphabet_size))

'''pi = np.full((k), 1/k)
m = np.full((k,k), 1/k)
e = np.full((k,alphabet_size), 1/alphabet_size)'''

pi = np.random.uniform(0, 1, (k))
pi /= pi.sum()

m = np.random.uniform(0, 1, (k,k))
for i in range(k):
	m[i] /= m[i].sum()

e = np.random.uniform(0, 1, (k,alphabet_size))
for i in range(k):
	e[i] /= e[i].sum()

t = 0
p = 0
while True:
	t += 1
	alpha = np.zeros((L,k))
	F = np.zeros((L, k))
	beta = np.zeros((L,k))
	gamma = np.zeros((L,k))
	xi = np.zeros((L,k,k))

	xi_prime = np.zeros((L))

	for i in range(k):
		alpha[0][i] = pi[i] * e[i][X[0]]
		F[0][i] = 0
		beta[L - 1][i] = 1
	F[0][0] = 1
	for l in range(1, L):
		for i in range(k):
			s = 0
			F_s = 0
			for j in range(k):
				s += alpha[l - 1][j] * m[j][i]
				F_s += F[l - 1][j] * m[j][i]
			alpha[l][i] = e[i][X[l]] * s
			F[l][i] = e[i][X[l]] * F_s
		alpha[l] /= alpha[l].sum()
	for l in range(L - 2, -1, -1):
		for i in range(k):
			s = 0
			for j in range(k):
				s += beta[l+1][j] * m[i][j] * e[j][X[l+1]]
			beta[l][i] = s
		beta[l] /= beta[l].sum()
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
	new_p = F[L - 1].sum()
	if new_p > 0 and new_p <= p:
		break
	print(t, p)
	p = new_p

print(p)