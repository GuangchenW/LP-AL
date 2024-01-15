import numpy as np

def generate_data(generator, n):
	out = []
	for i in range(n):
		out.append(generator())
	return np.array(out)

def nonlinear_oscillator():
	m = np.random.normal(1, 0.05)
	c1 = np.random.normal(1, 0.1)
	c2 = np.random.normal(0.1, 0.01)
	r = np.random.normal(0.5, 0.05)
	F1 = np.random.normal(1, 0.2)
	t1 = np.random.normal(1, 0.2)
	return [c1,c2,m,r,t1,F1]

def cantilever_tube():
	t = np.random.normal(5, 0.1)
	d = np.random.normal(42, 0.5)
	L1 = np.random.uniform(119.75, 120.25)
	L2 = np.random.uniform(59.75, 60.25)
	F1 = np.random.normal(3000, 300)
	F2 = np.random.normal(3000, 300)
	P = np.random.gumbel(30000, 3000)
	T = np.random.normal(90000, 9000)
	S_y = np.random.normal(220, 22)
	return [t,d,L1,L2,F1,F2,P,T,S_y]

if __name__ == "__main__":

	N_mcs = 10**6
	generator = cantilever_tube
	data = generate_data(generator, N_mcs)
	filename = generator.__name__ + ".npy"

	with open(filename, "wb") as f:
		np.save(f, data)
	print("done")