import nltk
import re
from nltk import FreqDist
from sklearn.cluster import KMeans, MiniBatchKMeans

## filter the alphabet
def alpha_filter(w):
  # pattern to match a word of non-alphabetical characters
    pattern = re.compile('^[^a-z]+$')
    if (pattern.match(w)):
        return True
    else:
        return False

file = open("foods.txt", 'rb')
contents=[]
count = 0
## read file from foods.txt
while 1:
    content = ""
    line = file.readline()
    str = line.decode('utf-8')
    count = count+1
    if not line:
        break
    if "review/text:" in str:
        content = content + str
        contents.append(content)

unique_L = []
each_L = []
W = []
for one in contents:
    tokens = nltk.word_tokenize(one) ## tokenize words
    words = [w.lower() for w in tokens] ## lower the words
    words = [w for w in words if not alpha_filter(w)]   ## filter the words
    each_L.append(words)
    words1 = sorted(set(words))  ## unique words
    for m in words1:
        unique_L.append(m)
    ## filter stopwords
    fstop = open('stopwords_CIS563.txt', 'r')
    stoptext = fstop.read()
    fstop.close()
    stopwords = nltk.word_tokenize(stoptext)
    stopwords.extend(["'s", "n't", "review/text"])
    stop_W = [w for w in words if w not in stopwords]
    for i in stop_W:
        W.append(i)

fdist = FreqDist(W) ## calculate the frequences
topkeys = fdist.most_common(500) ## top 500 words

## vectorize the words
vectorize = []
for m in each_L:
    one_ve = []
    for n in topkeys:
        count = 0
        for l in m:
            if l == n[0]:
                count = count + 1
        one_ve.append(count)
    vectorize.append(one_ve)

print(topkeys)

# run kmeans on vectorized review text
k_means = MiniBatchKMeans(n_clusters = 10)
k_means.fit(vectorize)
label_pred = list(k_means.labels_)
centroids = k_means.cluster_centers_

store_count = []
for m in range(0, 10):
    one = []
    for n in range(0, 500):
        one.append(0)
    store_count.append(one)
num = 0
for m in vectorize:
    count = 0
    for n in m:
        store_count[label_pred[num]][count] += n
        count = count + 1
    num = num + 1
centroid = k_means.cluster_centers_  ## centroid of each k-means cluster
res = []
store = []
m = 0
## find top 5 words representing each cluster and their feature values
for j in centroid:
    j = list(j)
    temp = []
    for i in range(5):
      temp.append((j.index(max(j)),max(j)))  ## do find the top 5 words
      j[j.index(max(j))] = m
    store.append(temp)
for j in store:
    one = []
    for l in j:
        one.append(topkeys[l[0]])
    res.append(one)
print(res)
