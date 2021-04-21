import numpy as np

# This whole thing doesn't work

# TODO:
	# Try researching Baum-Welch intialisation
	# Test it
	# How to generate test sequences?
		# Currently the only "random" option I have that actually gives a non-zero (no underflow) probability for L = 1000 is creating a random Markov Chain (i.e. alphabet_size states, each state determininstically emits a symbol)
		# Could just do patterns and stuff (but that's a bit boring)
		# Could try the method of random HMM and just leave it for ages
		# Could email him and ask for advice

L = 100
alphabet_size = 4

#X = np.random.randint(0, alphabet_size, (L), dtype=int)
#X = np.array([i % alphabet_size for i in range(L)])

X = np.zeros((L), dtype=int)

'''r = 2
X = []
while len(X) < L:
	for j in range(alphabet_size):
		for _ in range(r):
			X.append(j)
X = np.array(X[:L], dtype=int)'''

'''chain_pi = np.random.uniform(0, 1, (alphabet_size))
chain_pi /= chain_pi.sum()

chain_m = np.random.uniform(0, 1, (alphabet_size, alphabet_size))
for i in range(alphabet_size):
	chain_m /= chain_m.sum()

r = np.random.uniform(0, 1)
state = -1
for i in range(alphabet_size):
	if r <= chain_pi[i]:
		state = i
		break
	r -= chain_pi[i]

for l in range(L):
	X[l] = state
	r = np.random.uniform(0, 1)
	for i in range(alphabet_size):
		if r <= chain_m[state][i]:
			state = i
			break
		r -= chain_m[state][i]

p = chain_pi[X[0]]
for l in range(1, L):
	p *= chain_m[X[l - 1]][X[l]]

print("OG probability:", p)'''

'''k = 10
prob = 0.9

r = np.random.randint(0, k)
test_pi = np.random.uniform(0, 1, (k))
test_pi[r] = 0
test_pi /= test_pi.sum()/(1 - prob)
test_pi[r] = prob

test_m = np.zeros((k, k))
for i in range(k):
	r = np.random.randint(0, k)
	test_m[i] = np.random.uniform(0, 1, (k))
	test_m[i][r] = 0
	test_m[i] /= test_m[i].sum()/(1 - prob)
	test_m[i][r] = prob

test_e = np.zeros((k, alphabet_size))
for i in range(k):
	r = np.random.randint(0, alphabet_size)
	test_e[i] = np.random.uniform(0, 1, (alphabet_size))
	test_e[i][r] = 0
	test_e[i] /= test_e[i].sum()/(1 - prob)
	test_e[i][r] = prob

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

F = np.zeros((L, k))
F[0][0] = 1
for l in range(1, L):
	for i in range(k):
		F_s = 0
		for j in range(k):
			F_s += F[l - 1][j] * test_m[j][i]
		F[l][i] = test_e[i][X[l]] * F_s

print("OG probability:", F[L - 1].sum())'''

k = 10
num = 10
test_pi = np.zeros((k))
for i in range(num):
	test_pi[np.random.randint(k)] = np.random.uniform()
test_pi /= test_pi.sum()

test_m = np.zeros((k, k))
for i in range(k):
	for j in range(num):
		test_m[i][np.random.randint(k)] = np.random.uniform()
	test_m[i] /= test_m[i].sum()

test_e = np.zeros((k, alphabet_size))
for i in range(k):
	for j in range(num):
		test_e[i][np.random.randint(alphabet_size)] = np.random.uniform()
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

F = np.zeros((L, k))
for i in range(k):
	F[0][i] = test_pi[i] * test_e[i][X[0]]

for l in range(1, L):
	for i in range(k):
		F_s = 0
		for j in range(k):
			F_s += F[l - 1][j] * test_m[j][i]
		F[l][i] = test_e[i][X[l]] * F_s

print("OG probability:", F[L - 1].sum())

min_states = 5
max_states = 15

pi = np.random.uniform(0, 1, (max_states))
pi /= pi.sum()

m = np.random.uniform(0, 1, (max_states,max_states))
for i in range(max_states):
	m[i] /= m[i].sum()

e = np.random.uniform(0, 1, (max_states,alphabet_size))
for i in range(max_states):
	e[i] /= e[i].sum()

initial = True
last_random = True
for k in range(max_states, min_states - 1, -1):
	print("k =", k)
	t = 0
	p = 0
	stuck = False
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
			F[0][i] = pi[i] * e[i][X[0]]
			beta[L - 1][i] = 1
		#F[0][0] = 1
		for l in range(1, L):
			for i in range(k):
				s = 0
				F_s = 0
				for j in range(k):
					s += alpha[l - 1][j] * m[j][i]
					F_s += F[l - 1][j] * m[j][i]
				alpha[l][i] = e[i][X[l]] * s
				F[l][i] = e[i][X[l]] * F_s
			if alpha[l].sum() == 0:
				alpha[l] = np.ones((k))
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
		print("\t", t, p)
		if (new_p > 0 and new_p <= p) or t >= 35:
			break
		p = new_p
	if initial or p >= best_p:
		initial = False
		best_p = p
		best_pi = pi.copy()
		best_m = m.copy()
		best_e = e.copy()
		if p >= 1:
			break
	if last_random or p >= prev_p:
		print("Removing least probable")
		last_random = False
		# Remove least probable state
		least_probable = -1
		row = -1
		for i in range(k):
			s = 0
			for j in range(k):
				s += m[j][i]
			if least_probable == -1 or s < least_probable:
				least_probable = s
				col = i
		#col = np.unravel_index(np.argmin(m), m.shape)[0]
		m = np.delete(m, col, axis=0)
		m = np.delete(m, col, axis=1)
		e = np.delete(e, col, axis=0)
		pi = np.delete(pi, col)
	else:
		print("Generating randomly")
		last_random = True
		pi = np.random.uniform(0, 1, (k-1))
		pi /= pi.sum()

		m = np.random.uniform(0, 1, (k-1,k-1))
		for i in range(k-1):
			m[i] /= m[i].sum()

		e = np.random.uniform(0, 1, (k-1,alphabet_size))
		for i in range(k-1):
			e[i] /= e[i].sum()
	prev_p = p

print("best likelihood:", best_p)
print("best number of states:", len(best_pi))