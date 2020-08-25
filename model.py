import random, copy, collections, math, pickle
from sklearn.model_selection import train_test_split

class Model:
	"""
	data: array of tuples ('label', observation obj)
	observation obj: dict of observation category to value
	Ex. {'age': 20, 'GED': true, 'words': ['you', 'can', 'use', 'an', 'array', 'for', 'multiple', 'observations', 'per', 'category']}
	"""
	def __init__(self, data=[]):
		self.data = data
		self.labels = None
		self.categories = None
		self.training = None
		self.testing = None
		self.testing_result = None
		self.model = None

	def observe(self, observation):
		if type(observation) is list:
			self.data.extend(observation)
		else:
			self.data.append(observation)

	def process(self):
		self.labels = set()
		self.categories = set()
		for (label, observation) in self.data:
			self.labels.add(label)
			for category in observation:
				self.categories.add(category)
		self.training, self.testing = train_test_split(self.data)  

	def train(self):
		self.model = {}
		for category in self.categories:
			self.model[category] = {}
		for (label, observation) in self.training:
			for category, value in observation.items():
				if type(value) is list:
					for token in value:
						self.model[category].setdefault(token, collections.Counter())
						self.model[category][token][label] += 1
				else:
					self.model[category].setdefault(value, collections.Counter())
					self.model[category][value][label] += 1

	def test(self):
		correct = 0
		for (label, observation) in self.testing:
			prediction = self.predict(observation)
			if label == prediction:
				correct += 1
		acc = correct / len(self.testing)
		return {'total': len(self.testing), 'accuracy': acc, 'stddev': math.sqrt(acc * (1 - acc) / len(self.testing))}

	def save(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self.__dict__, f)

	def load(self, filename):
		with open(filename, 'rb') as f:
			tmp = pickle.load(f)
		self.__dict__.update(tmp)

	def predict(self, observation):
		probabilities = self.predict_probabilities(observation)
		return max(probabilities, key=probabilities.get)

	def predict_probabilities(self, observation):
		if self.model is None:
			raise RuntimeError('model was not trained before predicting')
		probabilities = {}
		for label in self.labels:
			probabilities[label] = 1
		for category, value in observation.items():
			if type(value) is list:
				for token in value:
					probs = self._get_probs(category, token)
					for label in probs:
						probabilities[label] *= probs[label]
			else:
				probs = self._get_probs(category, value)
				for label in probs:
					probabilities[label] *= probs[label]
		return probabilities

	def _get_probs(self, category, value):
		probabilities = {}
		if value not in self.model[category]:
			for label in self.labels:
				probabilities[label] = 1
			return probabilities
		total = sum(self.model[category][value].values()) + len(self.model[category])
		for label in self.labels:
			matches = self.model[category][value][label] + 1
			probabilities[label] = matches / total
		return probabilities

				


