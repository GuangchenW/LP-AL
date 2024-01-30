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

if __name__ == "__main__":

	output_directory = get_data_directory()

	N_mcs = 10**5
	generator = nonlinear_oscillator
	data = generate_data(generator, N_mcs)
	file_path = output_directory + generator.__name__ + ".npy"

	with open(file_path, "wb") as f:
		np.save(f, data)

	print("Done")