# -*- coding: utf-8 -*-
import proj1_data_loading as p1

redcoms = p1.RedditComments()
args = {
'n0':1e-5,
'beta':1e-4,
'epsilon':1e-6,
'maxiters':10
}
# Create object to load comments, perform linear regression
redcoms = RedditComments()
redcoms.load()


# Preprocess text, inputs into integers to be formatted into data matrix
text_train, X_train, y_train = redcoms.preprocess(redcoms.data[:10000])
text_val, X_val, y_val = redcoms.preprocess(redcoms.data[10000:11000])
text_test, X_test, y_test = redcoms.preprocess(redcoms.data[11000:12000])

# Find most frequent words in text and save to zip file
#    most_common_words = redcoms.most_common(text_train)
#    file = open('words.txt','w') 
#    for word_count in most_common_words:
#        file.write(word_count[0].ljust(10) + "\t" + word_count[1] + "\n")          
#    file.close() 

   # keywords = kfeatures.stopword_clean(kfeatures.data)
   # swearwords = sfeatures.profanity(sfeatures.data)

   

##############################################################################################


################################### task 3 - ealuate data ####################################

# No text Features.        

weights, error, X_train = redcoms.run_regression( closed_form=False, text_features=False, swear_words=True, size_comment=True,common_word_count=True, X=X_train, y=y_train, text=text_train, args=args )    
X_val = redcoms.add_features(X_val, text_val, closed_form=False, text_features=False, swear_words=True, size_comment=True,common_word_count=True);
mse, squared_error_array = redcoms.get_mse(X_val, y_val, weights)
print("Closed Form mean valid squared error:    " + str(mse[0]));
