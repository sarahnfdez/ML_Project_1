#!/usr/bin/python3

import json 
import collections
import numpy as np
import argparse
import string
import time
import nltk
from nltk.corpus import stopwords

class RedditComments:
    
    def __init__(self):
        self.data = None
        self.myname = self

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
#        for point in newd:
#            point['text'] = point['text'].lower()
        
        words = []
        for point in newd:
            text = point['text']
            for word in text:
                words.append(word)
        
        stop = stopwords.words('english')+list(string.punctuation)
        filtered_comments = []
        for w in words:
            if w not in stop:
                filtered_comments.append(w)        
        
        most_comm = collections.Counter(filtered_comments).most_common(171)
        most_common = []
        max_count = 0 #the maximum number of these 
        for tup in most_comm:
            word = tup[0]
            if (tup[1] > max_count):
                max_count = tup[1]
            most_common.append(word)
        most_common = most_common[11:]
        
        commonalities = []
        for point in newd:
            xcounts = [0]*160
            ind = 0
            text = point['text']
            for word1 in most_common:
                for word2 in text:
                    if word2 == word1:
                        xcounts[ind] += 1
                xcounts[ind] = xcounts[ind]
                ind += 1
            commonalities.append(xcounts)
        
         #scale array
        d_commonalities = np.reshape(commonalities, len(commonalities)*len(commonalities[0]) )
        maximum = np.max(d_commonalities)
        minimum = np.min(d_commonalities)
        i = 0
        for row in commonalities:
            j = 0
            for column in row:
                commonalities[i][j] = 6*(column - minimum)/(maximum - minimum)
                j+=1
            i+=1
        
        
        return commonalities
    def disc_profanity(self, swords):
   
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
        badwordscount = [0]*len(swords)
        ind = 0
    
        for point in swords:
            text = point['text']
            for word1 in text:
                for word2 in badwords:
                    if word1 == word2:
                        badwordscount[ind] = 1
                        break
            ind += 1
        
        return badwordscount
    def profanity(self, swords):
   
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
        badwordscount = [0]*len(swords)
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
        max_count = 0
        for tup in most_comm:
            word = tup[0]
            if(tup[1] > max_count):
                max_count = tup[1]
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
                xcounts[ind] = xcounts[ind]
                ind += 1
            commonalities.append(xcounts)
        
        #scale array
        d_commonalities = np.reshape(commonalities, len(commonalities)*len(commonalities[0]) )
        maximum = np.max(d_commonalities)
        minimum = np.min(d_commonalities)
        i = 0
        for row in commonalities:
            j = 0
            for column in row:
                commonalities[i][j] = 6*(column - minimum)/(maximum - minimum)
                j+=1
            i+=1
            
            
        

        return commonalities
   
    def most_common(self, comments):
        """
        finds the most common words with their word count from the list of comments
        """
        words = []
        for point in comments:
            text = point['text']
            for word in text:
                words.append(word)
    
        most_comm = collections.Counter(words).most_common(160)
        most_common = []
        count = []
        for tup in most_comm:
            word = tup[0]
            counter = tup[1]
                
            most_common.append(word)
            count.append(counter)
        word_list = np.array(most_common)
        count = np.array(count)
        word_list = np.reshape(word_list, [len(word_list), 1])
        count = np.reshape(count, [len(word_list), 1])
        word_list = np.append(word_list, count, axis=1)
        word_list = np.reshape(word_list, [len(count), 2])
        return word_list
                
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
        XT = X.transpose()
        XTdotX = XT.dot(X)
        Xtdoty = XT.dot(y)
        while True:
            if i == maxiters:
                i -= 1
                break
            alpha = n0 /(1 + beta[i])
            w[i] = w[i-1] - (2 * alpha * (XTdotX.dot(w[i-1]) - Xtdoty))
            if np.linalg.norm(w[i] - w[i-1], ord=2) <= epsilon:
                break

            
            i += 1
        ret = w[i]
        return ret
    
    def append_commonalities(self, commonalities, X_train):
        """
        input: x_train array without text data, and the text data we wish to append. 
        output: x_train data with text data
        """
        commonalities_arr = np.array(commonalities)
        x_train_commonalities = np.append(X_train, commonalities, axis=1)
        return x_train_commonalities
    def append_swear_count(self, X_train, text_train):
        """
        input: x_train array without text data, and the swear count we wish to append. 
        output: x_train data with text data
        """
        zero_arr = np.zeros([X_train.shape[0], 1])
        X_train = np.append(X_train, zero_arr, axis=0)
        X_train[-1, :] = self.profanity(text_train)
        return X_train
    
    def get_mse_crossv(self, X_data, Y_data, w):
       """
       input: 
           X_data: the data containing all the features we are looking at for all the instances
           Y_data: the true value of the output for each instance.
           weights: The weights calulated according to gradient descent or closed form solution matchin
                    the x_data to the y_data
                    
       output:
           mean_squared_error: predicts the y value for each x instance, compares to real y value and finds mean squared error between them.
       """
       num_instances = len(Y_data)
       i = 0;
       prediction_array = np.zeros(len(Y_data))
       
       ## find predicted value for each instance and store in array prediction_array
       for instance in X_data:
           j = 0;
           prediction = 0
           for feature in instance:
               prediction += feature*w[j]
               j += 1
           prediction_array[i] = prediction
           i +=1;
       
       mean_squared_error = 0
       # goes through each prediction to compare with try y value.
       squared_error_array = [pow(prediction_array[i] - Y_data[i], 2)[0] for i in range(num_instances)]
       squared_error_array = np.array(squared_error_array)
       for instance in squared_error_array:
           mean_squared_error += instance
       mean_squared_error = mean_squared_error/num_instances
       return mean_squared_error, squared_error_array;
   
    def size_comment(self, text):
        """
        returns a numpy array where each entry corresponds to the size of the comment it represents
            
        """
        comments = text
        comment_arr = np.zeros([len(text), 1])
        i = 0
        for comment in comments:
            this_comment = comment['text']
            size_comment = len(this_comment)
            comment_arr[i] = size_comment
            i+=1
        maximum = max(comment_arr)
        mean = np.mean(comment_arr)
        comment_arr = [-pow((comment_arr[i]-mean)/mean, 1) for i in range(len(comment_arr))]
        
        return comment_arr
    
    def common_count(self, comments):
        """ 
        goes through all the the comments, and creates an array that contains the number of most common words 
        in that comment
        
        """
        
        #############CALCULATE MOST COMMON WORDS WITHOUT STOP WORDS
        words = []
        for point in comments:
            text = point['text']
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
        
        count_arr = np.zeros([len(comments), 1])
        i = 0
        for comment in comments:
            count = 0
            comment_text = comment['text']
            for word in comment_text: 
                for common_word in most_common:
                    if (word==common_word):
                        count+=1
            count_arr[i] = count
            i+=1
        max_value = max(count_arr)
        return 10*count_arr/max_value 
                
                
    
    def run_regression(self, closed_form, text_features , swear_words, size_comment, common_word_count, **kwargs):
        """
        Runs linear regression according to the parameters above.
        
        Input:
            closed_form: boolean - if true, runs closed_form regression. 
            
            text_features: if true, appends text features to the regression.
            
             swear_words: if true, adds the swear_words to the regression features
            
            size_comments: appends the size of the comment feature.
            
            common_word_count: appends the number of text_features exist in each comment, to each comment
            kwargs:
                
      
                X, y : training and target data.
                
                args: arguments for regression, if they are being used.
                
                text: text features to append if needed.
        Output: 
            returns weights, mean_squared_error
            
        """
        time_start = time.time() #gives the current time in seconds since the start of 1970
        X_train = kwargs['X']
        y_train = kwargs['y']
        text_train = kwargs['text']
        #evaluate closed form regression
        if ( text_features == True ):
            #Cannot use text features with closed form. If attempted, will perform it with most_common_count
            if(closed_form == False):
                commonalities = self.stopword_clean(text_train)
                X_train = self.append_commonalities(commonalities, X_train)
            else:
                #must use common word count with this to avoid singular matrix. Make sure to use the count with the validation. 
                common_counts = self.common_count(text_train)
                X_train = np.append(X_train, common_counts, axis=1)
        if(common_word_count == True):
                common_counts = self.common_count(text_train)
                X_train = np.append(X_train, common_counts, axis=1)
        if (swear_words == True):
            swear_count = np.array(self.profanity(text_train))
            swear_count = np.reshape(swear_count, [X_train.shape[0], 1])
            X_train = np.append(X_train, swear_count, axis=1)
        if(size_comment==True):
            size_comments = self.size_comment(text_train)
            X_train = np.append(X_train, size_comments, axis=1)
            
           
        if (closed_form == True):
            c_w = self.closed_form(X_train, y_train)
            c_mean_squared_error, c_train_squared_error_array = self.get_mse(X_train, y_train, c_w)
           # c_valid_mean_squared_error, c_valid_squared_error_array = redcoms.get_mse(X_val, y_val, c_w)
            time_end = time.time()
            training_time = time_end - time_start
            print("Closed form training time: " + str(training_time))
            print("Closed form mean training squared error: " + str(c_mean_squared_error))
            
            return c_w, c_mean_squared_error, X_train, training_time
         #evaluate sgd regression
        else:
           args = kwargs['args'] 
           w0 = np.zeros([X_train.shape[1]])
           b = [args['beta']] * args['maxiters']
           sgd_w = self.gradient_descent(X_train, y_train, w0, b, args['n0'], args['epsilon'], args['maxiters'])
           sgd_mean_squared_error, sgd_train_squared_error_array = self.get_mse(X_train, y_train, sgd_w)
           time_end = time.time()
           training_time = time_end - time_start
           print("SGD  training time: " + str(training_time))
           print("SGD mean training squared error: " + str(sgd_mean_squared_error[0]))
           
           return sgd_w, sgd_mean_squared_error, X_train, time_end - time_start
    
    def get_mse(self, X_data, Y_data, w):
       """
       input: 
           X_data: the data containing all the features we are looking at for all the instances
           Y_data: the true value of the output for each instance.
           weights: The weights calulated according to gradient descent or closed form solution matchin
                    the x_data to the y_data
                    
       output:
           mean_squared_error: predicts the y value for each x instance, compares to real y value and finds mean squared error between them.
       """
       num_instances = len(Y_data)
       i = 0;
       prediction_array = np.array(Y_data)
       
       ## find predicted value for each instance and store in array prediction_array
       for instance in X_data:
           j = 0;
           prediction = 0
           for feature in instance:
               prediction += feature*w[j]
               j += 1
           prediction_array[i] = prediction[0]
           i +=1;
       
       mean_squared_error = 0
       # goes through each prediction to compare with try y value.
       squared_error_array = [pow(prediction_array[i] - Y_data[i], 2) for i in range(num_instances)]
       squared_error_array = np.array(squared_error_array)
       for instance in squared_error_array:
           mean_squared_error += instance
       mean_squared_error = mean_squared_error/num_instances
       return mean_squared_error, squared_error_array;
   
    def add_features(self, X, text_val, closed_form, text_features, swear_words, size_comment, common_word_count ):
        """
        Adds features to the X data if we added them in training. 
            Input:
                X data to which you want to add features
                Text data 
                Boolean:
                    text_features
                    swear_words
                    size_words
            Output:
                modified X data
        """
        if ( text_features == True ):
             if(closed_form == False):
                 commonalities = self.stopword_clean(text_val)
                 X = self.append_commonalities(commonalities, X)
             else:
                 common_counts = self.common_count(text_val)
                 X  = np.append(X, common_counts, axis=1)
        if(common_word_count == True):
               common_counts = self.common_count(text_val)
               X = np.append(X, common_counts, axis=1)      
        if (swear_words == True):
            swear_count = np.array(self.profanity(text_val))
            swear_count = np.reshape(swear_count, [X.shape[0], 1])
            X = np.append(X, swear_count, axis=1)
        if (size_comment==True):
            size_comments = self.size_comment(text_val)
            X = np.append(X, size_comments, axis=1)
        return X
    def preprocess_features(self, data, **kwargs):
        """
            Preprocesses the data. Converts text to all lowercase, splits into tokens.
            If any keyword args exist, preprocess the data according to them
            

            Input: data - data to preprocess (dict)
                    kwargs - dictionary of booleans that determine any preprocessing we wish to do with the data
            
            Output: data - preprocessed data
                    X - data matrix with bias (first column all 1s)
                    y - output vector 
        """
        text_features = False
        stop_words = False
        most_common = 1
        swear_words = False
        discrete_swear_words = False
        comment_length = False
        binary_text = False
        if 'text_features' in kwargs:
            text_features = kwargs['text_features']
            most_common_len = kwargs['most_common']
        if text_features and 'stop_words' in kwargs:
            stop_words = kwargs['stop_words']
        
        if 'swear_words' in kwargs:
            swear_words = kwargs['swear_words']
        if 'discrete_swear_words' in kwargs:
            discrete_swear_words = kwargs['discrete_swear_words']
            
        if 'comment_length' in kwargs:
            comment_length = kwargs['comment_length']
        if 'binary_text' in kwargs:
            binary_text = kwargs['binary_text']
        
       
        ## word list for if we want text features
        words = []
       
        for instance in data:
            if not stop_words:
                instance['text'] = instance['text'].lower().split()
            else:
                tokenizer = nltk.tokenize.TweetTokenizer()
                instance['text'] = tokenizer.tokenize(instance['text'].lower())
            if (stop_words): #if stop words is true, filter out the stop words from the comments 
                stop_list = stopwords.words('english')+list(string.punctuation)
                filtered_comments = []
                for w in instance['text']:
                    if w not in stop_list:
                        filtered_comments.append(w)
                instance['text'] = filtered_comments
        
                        
            if text_features or binary_text:
                #append word to words if we want text_features
                for word in instance['text']:
                    words.append(word)
            if instance['is_root']:
                instance['is_root'] = 1
            else:
                instance['is_root'] = 0
        
        if (text_features or binary_text):
            #most_comm is a list of the most common words    
            most_comm = collections.Counter(words).most_common(most_common_len)
            dictionary =[ most_comm[i][0] for i in range(len(most_comm))]
            commonalities =[]
        if comment_length:
            comment_length_arr = []
        if (comment_length or text_features or binary_text):
            for instance in data:
                if text_features or binary_text:
                    xcounts = [0]*most_common_len
                    text = instance['text']
                    index = 0
                    for word1 in dictionary:
                        for word2 in text:
                            if word2 == word1:
                                xcounts[index] += 1
                        xcounts[index] = xcounts[index]
                        index += 1
                    commonalities.append(xcounts)
                if comment_length:
                    comment_length_instance = len(instance['text'])
                    if comment_length_instance > 6:
                        comment_length_arr.append(1)
                    else: 
                        comment_length_arr.append(0)
#                    comment_length_arr.append(comment_length_instance)
                    
        if (text_features or binary_text):
            commonalities = np.array(commonalities, dtype='float')
            binary_commonalities = np.copy(commonalities)
            #scale commonalities array
            d_commonalities = np.reshape(commonalities, len(commonalities)*len(commonalities[0]) )
            maximum = np.max(d_commonalities)
            minimum = np.min(d_commonalities)
            mean = np.mean(d_commonalities)
            i = 0
            for row in commonalities:
                j = 0
                for column in row:
                    if binary_text:
                        binary_commonalities[i][j] = 1 if binary_commonalities[i][j] >= 1 else 0
                    else:
                        commonalities[i][j] = 6*(column - minimum)/(maximum-minimum)# (2*(column - minimum)/(maximum - minimum) - 1)
                    j+=1
                i+=1
    
        # Creates data matrix X and output vector y

        initial_point = data[0]
        X = np.array([[1, initial_point['is_root'], initial_point['controversiality'], initial_point['children']]])
        y = np.array([[initial_point['popularity_score']]])
        for i in range(1, len(data)):
            point = data[i]
            X = np.concatenate((X, [[1, point['is_root'], point['controversiality'], point['children']]]), axis=0)
            y = np.concatenate((y, [[point['popularity_score']]]), axis=0)
        X = X.astype('float')
        if text_features:
            X = np.append(X, commonalities, axis=1)
            
        if binary_text:
            X = np.append(X, binary_commonalities, axis=1)
        
        if swear_words:
            swear_count = np.array(self.profanity(data))
            swear_count = np.reshape(swear_count, [swear_count.shape[0], 1])
            X = np.append(X, swear_count, axis=1)
        if discrete_swear_words:
            swear_count = np.array(self.disc_profanity(data))
            swear_count = np.reshape(swear_count, [swear_count.shape[0], 1])
            X = np.append(X, swear_count, axis=1)
        if comment_length:
            comment_length_arr = np.array(comment_length_arr)
            comment_length_arr = np.reshape(comment_length_arr, [comment_length_arr.shape[0], 1])
            X = np.append(X, comment_length_arr, axis=1)
           
        return data, X, y
    def regress(self,closed_form, X_train, y_train, text, args):
        """
        Runs linear regression according to the parameters above.
        
        Input:
                closed_form: boolean - if true, runs closed_form regression. 
        
                X, y : training and target data.
                
                args: arguments for regression, if they are being used.
                
                text: text features to append if needed.
        Output: 
                returns weights, mean_squared_error, and regression compute time
            
        """
        time_start = time.time() #gives the current time in seconds since the start of 1970
              
        if (closed_form == True):  #evaluate closed form regression    
            c_w = self.closed_form(X_train, y_train)
            c_mean_squared_error, c_train_squared_error_array = self.get_mse(X_train, y_train, c_w)
            time_end = time.time()
            training_time = time_end - time_start            
            return c_w, c_mean_squared_error, training_time
        
        else: #evaluate sgd regression 
           w0 = np.zeros([X_train.shape[1]])
           b = [args['beta']] * args['maxiters']
           sgd_w = self.gradient_descent(X_train, y_train, w0, b, args['n0'], args['epsilon'], args['maxiters'])
           sgd_mean_squared_error, sgd_train_squared_error_array = self.get_mse(X_train, y_train, sgd_w)
           time_end = time.time()
           training_time = time_end - time_start
           
           return sgd_w, sgd_mean_squared_error, time_end - time_start
       
     
    def regress_crossv(self,closed_form, X_train, y_train, weights, text, args):
        """
        Runs linear regression according to the parameters above.
        
        Input:
                closed_form: boolean - if true, runs closed_form regression. 
        
                X, y : training and target data.
                
                args: arguments for regression, if they are being used.
                
                text: text features to append if needed.
        Output: 
                returns weights, mean_squared_error, and regression compute time
            
        """
        time_start = time.time() #gives the current time in seconds since the start of 1970
              
        if (closed_form == True):  #evaluate closed form regression    
            c_w = self.closed_form(X_train, y_train)
            c_mean_squared_error, c_train_squared_error_array = self.get_mse_crossv(X_train, y_train, c_w)
            time_end = time.time()
            training_time = time_end - time_start            
            return c_w, c_mean_squared_error, training_time
        
        else: #evaluate sgd regression 
           w0 = weights
           b = [args['beta']] * args['maxiters']
           sgd_w = self.gradient_descent(X_train, y_train, w0, b, args['n0'], args['epsilon'], args['maxiters'])
           sgd_mean_squared_error, sgd_train_squared_error_array = self.get_mse_crossv(X_train, y_train, sgd_w)
           time_end = time.time()
           training_time = time_end - time_start
           
           return sgd_w, sgd_mean_squared_error, time_end - time_start
 


if __name__ == "__main__":

    ###################################################################################
    # Parse the arguments, if any, to use for training (if not specified, resort to 
    # defaults)
    #    parser = argparse.ArgumentParser()
    #    parser.add_argument("--n0", help="hyperparameter in gradient descent, controls initial learning rate", default=1e-5)
    #    parser.add_argument("--beta", help="hyperparamter in gradient descent, controls speed of the decay", default=1e-4)
    #    parser.add_argument("--epsilon", help="error bound for gradient descent", default=1e-6)
    #    parser.add_argument("--maxiters", help="max iteration number for gradient descent", default=10)
    #    args = parser.parse_args()
    ###################################################################################
    args = {
    'n0':1e-5,
    'beta':1e-5,
    'epsilon':1e-9,
    'maxiters':100
    }
    # Create object to load comments, perform linear regression
    redcoms = RedditComments()
    redcoms.load()
    
    k_fold_cross_validation = True
    num_folds = 10
    closed_form = False
    text_features = True
    most_common = 160
    swear_words =False
    stop_words =  True
    discrete_swear_words = True
    comment_length = True
    binary_text = False
    
    if (not k_fold_cross_validation):
        # Preprocess text, inputs into integers to be formatted into data matrix
        text_train, X_train, y_train = redcoms.preprocess_features(redcoms.data[:10000], text_features=text_features, most_common=most_common, stop_words=stop_words, swear_words=swear_words, discrete_swear_words=discrete_swear_words, comment_length=comment_length, binary_text=binary_text)
        text_val, X_val, y_val = redcoms.preprocess_features(redcoms.data[10000:11000], text_features=text_features, most_common=most_common, stop_words=stop_words,  swear_words=swear_words, discrete_swear_words=discrete_swear_words, comment_length=comment_length, binary_text=binary_text)
        text_test, X_test, y_test = redcoms.preprocess_features(redcoms.data[11000:12000], text_features=text_features, most_common=most_common,stop_words=stop_words,  swear_words=swear_words, discrete_swear_words=discrete_swear_words, comment_length=comment_length, binary_text=binary_text)
        
            
        
        training_weights, training_error, training_time = redcoms.regress( closed_form, X_train, y_train, text_train, args )    
        validation_error, squared_error_array = redcoms.get_mse(X_test, y_test, training_weights)
        if closed_form:
            print("Closed form training time:   " + str(training_time))
            print("Closed form training mean squared error: " +  str(training_error[0]))
            print("closed form validation mean squared error:    " + str(validation_error));
        else:
            print("SGD training time:   " + str(training_time))
            print("SGD training mean squared error: " +  str(training_error[0]))
            print("SGD validation mean squared error:    " + str(validation_error));
        
    else:
       
       text, X, y = redcoms.preprocess_features(redcoms.data[1000:], text_features=text_features, most_common=most_common, stop_words=stop_words, swear_words=swear_words, discrete_swear_words=discrete_swear_words, comment_length=comment_length, binary_text=binary_text)
       text_test, X_test, y_test = redcoms.preprocess_features(redcoms.data[:1000], text_features=text_features, most_common=most_common, stop_words=stop_words, swear_words=swear_words, discrete_swear_words=discrete_swear_words, comment_length=comment_length, binary_text=binary_text)
     
        #first 1000 for testing, last 1100 for train/valid split.
        
       foldnum = num_folds
       seg_length = len(X)/num_folds
       
       
       tempX = np.copy( X)
       tempy = np.copy(y)
       #starting weights
       training_weights = np.zeros([X.shape[1], 1])
       mean_train_mse_arr = np.array([])
       mean_valid_mse_arr = np.array([])
       for i in range(foldnum-1):
           
           seg_length = int(seg_length)
           X_val = X[(i) * seg_length: (i+1)*seg_length]
           y_val = y[(i) * seg_length: (i+1)*seg_length]
           X_train = np.delete(tempX, [range((i) * seg_length,  (i+1)*seg_length)], axis=0)
           y_train = np.delete(tempy, [range((i) * seg_length,  (i+1)*seg_length)], axis=0)
           
           #get training weights, input last iteration's training weights. 
           training_weights, training_error, training_time = redcoms.regress_crossv( closed_form, X_train, y_train, training_weights, text, args )    
         #  training_weights = training_weights[:, 0]
           valid, squared_error_array = redcoms.get_mse_crossv(X_val, y_val, training_weights)
           mean_train_mse_arr = np.append( mean_train_mse_arr, training_error)
           mean_valid_mse_arr = np.append(mean_valid_mse_arr, valid)
         #  test_error, square_error_array = redcoms.get_mse(X_test, y_test, training_weights)
           print("Fold : " + str(i+1) + "\nTraining MSE: \t" + str(training_error) + "\nValidation MSE:\t : " + str(valid))
       
       
       print ("\nmean of training: " + str(np.mean(mean_train_mse_arr)))
       print("mean of valid :    " + str(np.mean(mean_valid_mse_arr)))
       
       
       test_err, squared_error_array = redcoms.get_mse_crossv(X_val, y_val, training_weights)
       print ("testing MSE:  " + str(np.mean(test_err)))
       
       
       
       
       