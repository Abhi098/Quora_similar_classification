import pandas as pd 
import numpy as np  
from fuzzywuzzy import fuzz
import gensim
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
from scipy.spatial.distance import cosine, cityblock, jaccard
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("train_quora.csv")
print(data.head())

data=data.drop(['id','qid1','qid2'],axis=1)

print(data.columns)

###FEATURE ENGINEERING###

data['q1_len']=data.question1.apply(lambda x: len(str(x)))##Length of Question1
data['q2_len']=data.question2.apply(lambda x: len(str(x)))##Length of Question2
data['diff_len']=data.q1_len-data.q2_len ##Difference of length

data['char_len_q1']=data.question1.apply(lambda x: len("".join(set(str(x).replace(" ","")))))##CHAR LENGTH
##str(x).replace will replace the space and then convert it into a set and the join will join all the elemnets in the set 
##and then find its length
data['char_len_q2']=data.question2.apply(lambda x: len("".join(set(str(x).replace(" ","")))))

data['word_len_q1']=data.question1.apply(lambda x: len(str(x).split()))##WORD LENGTH

data['word_len_q2']=data.question2.apply(lambda x: len(str(x).split()))

data['common_words']=data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))),axis=1)


print(data.columns)

data.to_csv("train1.csv")

data=pd.read_csv("train1_quora.csv")

print(data['common_words'])

data['fuzz_qratio']=data.apply(lambda x:fuzz.QRatio(str(x['question1']),str(x['question2'])),axis=1)

data['fuzz_Wratio']=data.apply(lambda x:fuzz.WRatio(str(x['question1']),str(x['question2'])),axis=1)


data['fuzz_partial_ratio']=data.apply(lambda x:fuzz.partial_ratio(str(x['question1']),str(x['question2'])),axis=1)



model = gensim.models.KeyedVectors.load_word2vec_format(
'GoogleNews-vectors-negative300.bin.gz', binary=True)

stop_words = set(stopwords.words('english'))

def sent2vec(s, model): 
	M = []
	words = word_tokenize(str(s).lower())
	for word in words:
		#It shouldn't be a stopword
		if word not in stop_words:
		#nor contain numbers
		if word.isalpha():
		#and be part of word2vec
		if word in model:
			M.append(model[word])
			M = np.array(M)
	
	if len(M) > 0:
		v = M.sum(axis=0)
		return v / np.sqrt((v ** 2).sum())
	else:
		return np.zeros(300)


w2v_q1 = np.array([sent2vec(q, model) 
                   for q in data.question1])
w2v_q2 = np.array([sent2vec(q, model) 
                   for q in data.question2])

data['cosine_distance'] = [cosine(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['cityblock_distance'] = [cityblock(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]
data['jaccard_distance'] = [jaccard(x,y) 
for (x,y) in zip(w2v_q1, w2v_q2)]

q1=['q1_len','q2_len','diff_len','char_len_q1','char_len_q2','word_len_q1','word_len_q2','common_words','fuzz_qratio''fuzz_Wratio','fuzz_partial_ratio','cosine_distance','cityblock_distance','jaccard_distance']

scaler = StandardScaler()
y = data.is_duplicate.values
y = y.astype('float32').reshape(-1, 1)
X = data[q1]
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
X = scaler.fit_transform(X)


np.random.seed(42)
n_all, _ = y.shape
idx = np.arange(n_all)
np.random.shuffle(idx)
n_split = n_all // 10
idx_val = idx[:n_split]
idx_train = idx[n_split:]
x_train = X[idx_train]
y_train = np.ravel(y[idx_train])
x_val = X[idx_val]
y_val = np.ravel(y[idx_val])

logres = linear_model.LogisticRegression(C=0.1, 
                                 solver='sag', max_iter=1000)
logres.fit(x_train, y_train)
lr_preds = logres.predict(x_val)
log_res_accuracy = np.sum(lr_preds == y_val) / len(y_val)
print("Logistic regr accuracy: %0.3f" % log_res_accuracy)
