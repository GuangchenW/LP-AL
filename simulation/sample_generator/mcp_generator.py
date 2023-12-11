import numpy as np

if __name__ == "__main__":

	N_mcs = 10**6

	out = []

	for i in range(N_mcs):
		omega = np.random.normal(1000, 100)
		L = np.random.normal(6, 0.9)
		b = np.random.normal(250, 37.5)
		out.append([omega, b, L])

	data = np.array(out)

	with open("data.npy", "wb") as f:
		np.save(f, data)
	print("done")

