#!/usr/bin/python3

import json 
import collections
import numpy as np
import argparse
import string
import nltk
from nltk.corpus import stopwords

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

    def stopword_clean(self, comments):
        """
            Cleans comments by removing all stopwords and punctuation
        
            Input: comments - preprocessed data
            Output: filtered_comments - comments with stopwords and 
                    punctuation removed
        
        """
        newd = comments
        for point in newd:
            point['text'] = point['text'].lower()
        
        words = []
        for point in newd:
            text = nltk.word_tokenize(point['text'])
            for word in text:
                words.append(word)
        
        stop = stopwords.words('english')+list(string.punctuation)
        filtered_comments = []
        for w in words:
            if w not in stop:
                filtered_comments.append(w)        
        
        most_comm = collections.Counter(filtered_comments).most_common(171)
        most_common = []
        for tup in most_comm:
            word = tup[0]
            most_common.append(word)
        most_common = most_common[11:]
        
        commonalities = []
        for point in newd:
            xcounts = [0]*160
            ind = 0
            text = nltk.word_tokenize(point['text'])
            for word1 in most_common:
                for word2 in text:
                    if word2 == word1:
                        xcounts[ind] += 1
                ind += 1
            commonalities.append(xcounts)
        
        return commonalities
    
    def profanity(self, swords):
        for point in swords:
            point['text'] = point['text'].lower()
            point['text'] = nltk.word_tokenize(point['text'])
    
    
        badwords = """apeshit ass assfuck asshole assbag assbandit assbang 
                assbanged assbanger assbangs assbite assclown asscock
                asscracker asses assface assfaces assfuck assfucker ass-fucker 
                assfukka asshat asshead asshole assholes asslick asslicker
                beotch biatch bitch bitchtit bitchass bitched bitches bitchin 
                bitching bitchtits bitchy hell bullshit bullshits bullshitted 
                buttfuck buttfucker cock cockmunch cockmuncher cocks cocksuck 
                cocksucked cocksucker cock-sucker cocksuckers cocksucking 
                cocksucks cunt cuntbag dammit damn damned damnit dick dickbag 
                dickface dickhead dickheads dickhole dickish dicks dipshit dong 
                dumass dumbass dumbasses Dumbcunt dumbfuck dumbshit dumshit 
                fatass fcuk fcuker fcuking fuc fuck fuckass fuck-ass fuckbag 
                fuckboy fucked fucked up fucker fuckers fuckface fuckhead 
                fuckheads fuckhole fuckin fucking fuckings fucks godamn godamnit 
                goddam god-dam goddammit goddamn goddamned god-damned goddamnit 
                ho hoe jackass jackasses mother fucker motherfucked motherfucker 
                motherfuckers motherfuckin motherfucking motherfuckings motherfucks 
                punkass pussies pussy pussys sex sexy shit shitface shithead 
                shitheads shithole shiting shitings shits slut sluts smartass 
                smartasses whore whores whoring"""
    
        badwords = nltk.word_tokenize(badwords)
        badwordscount = [0]*12000
        ind = 0
    
        for point in swords:
            text = point['text']
            for word1 in text:
                for word2 in badwords:
                    if word1 == word2:
                        badwordscount[ind] += 1
            ind += 1
        
        return badwordscount
    
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
    kfeatures = RedditComments()
    kfeatures.load()
    sfeatures = RedditComments()
    sfeatures.load()
    
    #redcoms.example()
    
    # Preprocess text, inputs into integers to be formatted into data matrix
    text_train, X_train, y_train = redcoms.preprocess(redcoms.data[:10000])
    text_val, X_val, y_val = redcoms.preprocess(redcoms.data[10000:11000])
    text_test, X_test, y_test = redcoms.preprocess(redcoms.data[11000:12000])

    # Find most frequent words in text
    commonalities = redcoms.freq_dist(text_train)
    keywords = kfeatures.stopword_clean(kfeatures.data)
    swearwords = sfeatures.profanity(sfeatures.data)

    w_train = redcoms.closed_form(X_train, y_train)
    print("Closed form weights: {}".format(w_train))
    w0 = np.array([[0],[0],[0],[0]])
    b = [args.beta] * args.maxiters
    w_train = redcoms.gradient_descent(X_train, y_train, w0, b, args.n0, args.epsilon, args.maxiters)
    print("Gradient descent weights: {}".format(w_train))

##############################################################################################

    
    
                
                
