import os
import numpy as np

class MCP_Manager:
	def __init__(self):
		self.directory = os.path.dirname(os.path.realpath(__file__))

	def get_data_directory(self, name):
		directory = self.directory+"/test_data/"
		if not os.path.exists(directory):
			os.makedirs(directory)
		data_directory = directory+name+".npy"
		return data_directory

	def generate_data(self, name, generator, n):
		out = []
		for i in range(n):
			out.append(generator())
		data = np.array(out)

		file_path = self.get_data_directory(name)
		with open(file_path, "wb") as f:
			np.save(f, data)

		print("Successfully generated Monte-Carlo population for %s." % name)

	def load_data(self, name):
		directory = self.get_data_directory(name)
		points = None
		with open(directory, 'rb') as file:
			points = np.load(file)
		return points

	def data_exists(self, name):
		directory = self.get_data_directory(name)
		return os.path.exists(directory)

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

	output_directory = get_data_directory()

	N_mcs = 10**5
	generator = nonlinear_oscillator
	data = generate_data(generator, N_mcs)
	file_path = output_directory + generator.__name__ + ".npy"

	with open(file_path, "wb") as f:
		np.save(f, data)

	print("Done")