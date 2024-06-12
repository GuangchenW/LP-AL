import sys
import math
import numpy as np
from sklearn.cluster import KMeans, k_means
from sklearn_extra.cluster import KMedoids

from .base_evaluator import BaseEvaluator

class KMeans_Batch(BaseEvaluator):
	def __init__(self, acq_func, logger=None):
		super().__init__(acq_func=acq_func, logger=logger)
		self.name="cluster"

	# TODO: Not very clean
	def set_grad(self, grad):
		self.grad = grad

	def obtain_batch(
		self,
		subset_points,
		mean,
		variance,
		doe_input,
		doe_response,
		n_points
	):
		batch = []

		k = min(n_points, len(subset_points))

		#kmedoids = KMedoids(n_clusters=k, random_state=0).fit(subset_points)
		#kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(subset_points)
		weights = variance/(mean**2)
		centroids, labels, _ = k_means(subset_points, n_clusters=k, sample_weight=weights, n_init="auto", random_state=0)

		'''
		for pnt in kmedoids.cluster_centers_:
			batch.append({
				"next": pnt,
				"mean": float("nan"),
				"variance": float("nan"),
				"utility": float("nan")
				})
		'''

		for pnt in centroids:
			batch.append({
				"next": pnt,
				"mean": float("nan"),
				"variance": float("nan"),
				"utility": float("nan")
				})
		'''
		for i in range(k):
			mask = kmeans.labels_==i
			candidates = subset_points[mask]
			c_mean = mean[mask]
			c_variance = variance[mask]
			utilities = self.acq_func.acquire(candidates, c_mean, c_variance, doe_input, doe_response)
			max_id = np.nanargmax(utilities)
			batch.append({
				"next": candidates[max_id],
				"mean": c_mean[max_id],
				"variance": c_variance[max_id],
				"utility": utilities[max_id]
				})
		'''

		return np.array(batch)

