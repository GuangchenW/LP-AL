import sys
import os

class Logger:
	def __init__(self, out_target=None, silence=False):
		directory = os.path.dirname(os.path.realpath(__file__))
		target_path = directory+"/test_logs/"
		if not os.path.exists(target_path):
			os.makedirs(target_path)
		self.out_target = open(target_path+out_target, "w") if out_target else sys.stdout
		self.silence = silence

	def log(self, content):
		self.out_target.write(content+"\n")
		if not self.silence and self.out_target is not sys.stdout:
			print(content)

	def log_batch(self, iteration, batch):
		self.log("**"*25)
		self.log("Iteration [%i] | Batch size [%i]" % (iteration, len(batch)))
		for candidate in batch:
			self.log("--"*25)
			self.log("Selected" + str(candidate["next"]))
			self.log("SCORE : %.3f | MEAN : %.4g | VARIANCE : %.4g" % (
				candidate["utility"],
				candidate["mean"],
				candidate["variance"]))
		self.log("**"*25)

	def clean_up(self):
		if self.out_target is not sys.stdout:
			out_target.close()

