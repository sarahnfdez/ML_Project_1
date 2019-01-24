#!/usr/bin/python3

import json 
import collections
import numpy as np
import argparse

class RedditComments:
	def __init__(self):
		self.data = None

	def load(self):
		""" 
			It a list of data points, where each datapoint is a dictionary 
			with the following attributes:
 			- popularity_score : a popularity score for this comment (based on the 
			number of upvotes) (type: float)
 			- children : the number of replies to this comment (type: int)
 			- text : the text of this comment (type: string)
 			- controversiality : a score for how "controversial" this comment is 
			(automatically computed by Reddit)
 			- is_root : if True, then this comment is a direct reply to a post; 
			if False, this is a direct reply to another comment 
		"""
		with open("proj1_data.json") as fp:
			self.data = json.load(fp)

	def example(self):
		"""
			Example provided in starter code, prints a single data point
		"""
		data_point = self.data[0] # select the first data point in the dataset
		# Now we print all the information about this datapoint
		for info_name, info_value in data_point.items():
			print(info_name + " : " + str(info_value))

	def preprocess(self, dd):
		"""
			Preprocesses the data. Converts text to all lowercase, splits into tokens

			Input: dd - data to preprocess (dict)
			Output: dd - preprocessed data
					X - data matrix with bias (first column all 1s)
					y - output vector 
		"""
		for point in dd:
			point['text'] = point['text'].lower().split()
			if point['is_root']:
				point['is_root'] = 1
			else:
				point['is_root'] = 0

		# Creates data matrix X and output vector y
		# The 1 represents x0 = 1, the bias term
		initial_point = dd[0]
		X = np.array([[1, initial_point['is_root'], initial_point['controversiality'], initial_point['children']]])
		y = np.array([[initial_point['popularity_score']]])
		for i in range(1, len(dd)):
			point = dd[i]
			X = np.concatenate((X, [[1, point['is_root'], point['controversiality'], point['children']]]), axis=0)
			y = np.concatenate((y, [[point['popularity_score']]]), axis=0)

		return dd, X, y

	def freq_dist(self, comments):
		"""
			Calculates frequencies of words in the text values of the given
			data dicts

			Input: comments - data points to extract frequency distribution from
			Output: commonalities - vectors representing count of most frequent words in
								a particular text
		"""
		words = []
		for point in comments:
			text = point['text']
			for word in text:
				words.append(word)

		most_comm = collections.Counter(words).most_common(160)
		most_common = []
		for tup in most_comm:
			word = tup[0]
			most_common.append(word)
		commonalities = []

		# Creates a 160-dim array for every comment whose values are the number of
		# times the frequent words are used
		for point in comments:
			xcounts = [0]*160
			ind = 0
			text = point['text']
			for word1 in most_common:
				for word2 in text:
					if word2 == word1:
						xcounts[ind] += 1
				ind += 1
			commonalities.append(xcounts)
		return commonalities

	def closed_form(self, X, y):
		"""
			Implements the closed-form solution w = (X^T X)^-1 X^T y, where X is
			the data matrix and y is the output vector (popularity score vector)

			Input: X - data matrix
				   y - target vector
			Output: w - weight vector (num features + 1 by 1)
		"""
		XT = X.transpose()
		intermediate = np.linalg.inv(XT.dot(X))
		w = intermediate.dot(XT).dot(y)
		return w

	def gradient_descent(self, X, y, w0, beta, n0, epsilon, maxiters):
		"""
			Steps through a function proportional to the negative of the gradient
			to find its local minimum. We want to produce a sequence of weight solutions
			such that the error is decreasing: Err(w0) > Err(w1) > Err(w2) ...
			The n0 hyperparameter is the initial learning rate and the beta hyperparameter
			controls the speed of the decay.

			Input: X - data matrix
				   y - target vector
				   w0 - initial weight vector 
				   beta, n0, epsilon - hyperparameters (beta is a vector)
				   maxiters - the maximum number of iterations if it won't converge
			Output: w - estimated weight vector (num features + 1 by 1)
		"""
		i = 1
		w = [0] * maxiters
		w[0] = w0
		while True:
			if i == maxiters:
				i -= 1
				break
			alpha = n0 /(1 + beta[i])
			XT = X.transpose()
			w[i] = w[i-1] - (2 * alpha * (XT.dot(X).dot(w[i-1]) - XT.dot(y)))
			if np.linalg.norm(w[i] - w[i-1], ord=2) <= epsilon:
				break
			i += 1
		return w[i]

if __name__ == "__main__":

	###################################################################################
	# Parse the arguments, if any, to use for training (if not specified, resort to 
	# defaults)
	parser = argparse.ArgumentParser()
	parser.add_argument("--n0", help="hyperparameter in gradient descent, controls initial learning rate", default=1e-5)
	parser.add_argument("--beta", help="hyperparamter in gradient descent, controls speed of the decay", default=1e-4)
	parser.add_argument("--epsilon", help="error bound for gradient descent", default=1e-6)
	parser.add_argument("--maxiters", help="max iteration number for gradient descent", default=10)
	args = parser.parse_args()
	###################################################################################

	# Create object to load comments, perform linear regression
	redcoms = RedditComments()
	redcoms.load()
	#redcoms.example()
	
	# Preprocess text, inputs into integers to be formatted into data matrix
	text_train, X_train, y_train = redcoms.preprocess(redcoms.data[:10000])
	text_val, X_val, y_val = redcoms.preprocess(redcoms.data[10000:11000])
	text_test, X_test, y_test = redcoms.preprocess(redcoms.data[11000:12000])

	# Find most frequent words in text
	commonalities = redcoms.freq_dist(text_train)

	w_train = redcoms.closed_form(X_train, y_train)
	print("Closed form weights: {}".format(w_train))
	w0 = np.array([[0],[0],[0],[0]])
	b = [args.beta] * args.maxiters
	w_train = redcoms.gradient_descent(X_train, y_train, w0, b, args.n0, args.epsilon, args.maxiters)
	print("Gradient descent weights: {}".format(w_train))
