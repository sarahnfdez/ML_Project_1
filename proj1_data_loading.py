import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import collections

with open("proj1_data.json") as fp:
    data = json.load(fp)
    
# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 

# Example:
data_point = data[0] # select the first data point in the dataset

# Now we print all the information about this datapoint
for info_name, info_value in data_point.items():
    print(info_name + " : " + str(info_value))

#preprocessing the data
def preprocessing(dd):
    for point in dd:
        point['text'] = point['text'].lower().split()
    return dd

training = preprocessing(data[:10000])
validation = preprocessing(data[10000:11000])
testing = preprocessing(data[11000:12000])

words = []
for point in data:
    text = point['text']
    for word in text:
            words.append(word)
            
most_comm = collections.Counter(words).most_common(160)
most_common = []
for tup in most_comm:
    word = tup[0]
    most_common.append(word)

commonalities = []
#creates a 160-dim array for every comment whose 
#values are the number of times the frequent words are used
for point in training:
    xcounts = [0]*160
    ind = 0
    text = point['text']
    for word1 in most_common:
        for word2 in text:
            if word2 == word1:
                xcounts[ind] = xcounts[ind]+1
        ind = ind+1
    commonalities.append(xcounts)
