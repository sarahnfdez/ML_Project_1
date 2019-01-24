#!/usr/bin/python3

import json 
import collections
from nltk.corpus import stopwords
import numpy as np

class RedditComments:
	def __init__(self):
		self.data = None
		self.processed_text = None

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
		"""
		for point in dd:
			point['text'] = point['text'].lower().split()
			if point['is_root']:
				point['is_root'] = 1
			else:
				point['is_root'] = 0
		self.processed_text = dd

		# create data matrix X and output vector y
		initial_point = dd[0]
		X = np.array([[initial_point['is_root'], initial_point['controversiality'], initial_point['children']]])
		y = np.array([[initial_point['popularity_score']]])
		for i in range(1, len(dd)):
			point = dd[i]
			X = np.concatenate((X, [[point['is_root'], point['controversiality'], point['children']]]), axis=0)
			y = np.concatenate((y, [[point['popularity_score']]]), axis=0)

		return X, y

	def freq_dist(self):
		"""
			Calculates frequencies of words in the text values of the given
			data dicts
		"""
		all_words = []
		for point in self.processed_text:
			text = point['text']
			for word in text:
				all_words.append(word)

		# removes stop words
		stop_words = set(stopwords.words('english')) 
		words = [ w for w in all_words if not w in stop_words]

		most_comm = collections.Counter(words).most_common(160)
		most_common = []
		for tup in most_comm:
			word = tup[0]
			most_common.append(word)
		commonalities = []

		# creates a 160-dim array for every comment whose values are the number of
		# times the frequent words are used
		for point in self.processed_text:
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
		"""
		XT = X.transpose()
		XTX_inverse = np.linalg.inv(XT.dot(X))
		w = XTX_inverse.dot(XT).dot(y)
		return w

if __name__ == "__main__":
	redcoms = RedditComments()
	redcoms.load()
	#redcoms.example()
	
	X_train, y_train = redcoms.preprocess(redcoms.data[:10000])
	X_val, y_val = redcoms.preprocess(redcoms.data[10000:11000])
	X_test, y_test = redcoms.preprocess(redcoms.data[11000:12000])

	commonalities = redcoms.freq_dist()

	w_train = redcoms.closed_form(X_train, y_train)
