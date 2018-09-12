import sys
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import csv
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack

################################## SETTINGS ##################################

#minumum and maximum n-gram size for textual features:
n_gram_min, n_gram_max = 1, 3

# Model: '1' (one-step 'Speech'), '2a' (two-step 'Motion-speech with' automa-
# tic motion classification), or '2b' (two-step with Govt./Opp motion labels)
model = sys.argv[1]

# Label – 'vote' or 'manual':
sentiment_label = sys.argv[2]

# Features:
# 0 = Textual features only
# 1 = Textual features + speaker party affiliation
# 2 = Textual features + speaker party affiliation + debate ID
# 3 = Textual features + speaker party affiliation + debate ID + motion party affiliation
# 4 = Speaker party affiliation + debate ID
# 5 = Speaker party affiliation + debate ID + motion party affiliation
features = int(sys.argv[3])

##############################################################################

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print('Welcome to ParliaSent!\n\nSpeech-level sentiment analysis for Hansard UK parliamentary debate transcripts.\n')

# Create custom stopwords:
hansard_stopwords = ({'friend', 'honourable', 'hon', 'gentleman', 'lady', 
					  'give', 'way', 'house', 'amendment', 'beg', 'move', 
					  'member', 'question', 'green', 'white', 'paper', 
					  'bill', 'statement', 'government', 'prime', 'minister', 
					  'opposition', 'party', 'mr', 'rose'})
sklhansard_stopwords = text.ENGLISH_STOP_WORDS.union(hansard_stopwords)
# lists of party names:
parties =  ['Con', 'Lab', 'LD', 'SNP', 'PC','SDLP', 'DUP', 'UUP', 'Green', 
			'Ind', 'IndCon', 'IndUU']
motion_parties = ['Con', 'Lab', 'LD', 'SNP', 'DUP']

n_gram_min, n_gram_max = 1, 3

# establish settings for text vectorizer and classifiers:
vectorizer = TfidfVectorizer(min_df=5, max_df = 1.0, 
	ngram_range=(n_gram_min,n_gram_max), sublinear_tf=True, use_idf =True, 
	stop_words=sklhansard_stopwords)
SVM = LinearSVC()
MLP = MLPClassifier(solver='lbfgs', max_iter=200, alpha=1e-5, hidden_layer_sizes=(100,), random_state=1)

# establish k-fold :
folds_no = 10
kf = StratifiedKFold(n_splits=folds_no, shuffle=True)

# initialize results:
SVM_motion_acc, SVM_motion_pres, SVM_motion_rec, SVM_motion_f1 = 0, 0, 0, 0
MLP_motion_acc, MLP_motion_pres, MLP_motion_rec, MLP_motion_f1 = 0, 0, 0, 0
SVM_speech_acc, SVM_speech_pres, SVM_speech_rec, SVM_speech_f1 = 0, 0, 0, 0
MLP_speech_acc, MLP_speech_pres, MLP_speech_rec, MLP_speech_f1 = 0, 0, 0, 0

# load debate data:
data = open('HanDeSeT.csv')
debates = list(csv.reader(data))[1:]

# empty lists for features:
X_motions = []
X_speeches = []
X_speeches_party = []
X_speeches_partyid = []
X_speeches_partyidmotion = []
X_speeches_govtG = [] 
X_speeches_govtO = []
X_speeches_party_govtG = [] 
X_speeches_party_govtO = []
X_speeches_partyid_govtG = []
X_speeches_partyid_govtO = []
X_speeches_partyidmotion_govtG = []
X_speeches_partyidmotion_govtO = []
X_speeches1 = [] 
X_speeches_auto0 = []
X_speeches_party_auto1 = [] 
X_speeches_party_auto0 = []
X_speeches_partyid_auto1 = []
X_speeches_partyid_auto0 = []
X_speeches_partyidmotion_auto1 = []
X_speeches_partyidmotion_auto0 = []
# empty lists for labels:
y_motions = []
y_speeches = []
y_speeches1 = []
y_speeches0 = []

# RUN ONE-STEP MODEL
if model == '1':
	two_step = False
	for row in debates:
		speech_len = 0
		speech_words = ''
		for utterance in row[6:11]: # utterancess 1-5
			for word in utterance.split():
				speech_len += 1
				speech_words = speech_words + word + ' '
		X_speeches.append(speech_words)
		# debate ID feature:
		deb_id_feat = np.zeros((1,104))[0]
		for debate_id in range(104):
			if int(row[0]) == debate_id:
				deb_id_feat[debate_id] = 1
		# party affiliation feature:
		party_feat = np.zeros((1,len(parties)))[0]
		for party in range(len(parties)):
			if row[13] == parties[party]:
				party_feat[party] = 1
		X_speeches_party.append(party_feat) 
		X_speeches_partyid.append(np.append(party_feat, deb_id_feat))
		# motion party feature:
		motion_party_feat = np.zeros((1,len(motion_parties)))[0]
		for party in range(len(motion_parties)):
			if row[5] == motion_parties[party]:
				motion_party_feat[party] = 1
		X_speeches_partyidmotion.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
		if sentiment_label == 'vote':
			sent_label = 'division vote;'
			y_speeches.append(int(row[12]))
		else:
			sent_label = 'manually annotated;'
			y_speeches.append(int(row[11]))
	y_speeches = np.array(y_speeches)
	data_combos = [((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'text features only'], 'no'),
			   ((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'text features + speaker party affiliation'], 'both', (X_speeches_party, X_speeches_party)),
			   ((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'text features + speaker party affiliation + debate ID'], 'both', (X_speeches_partyid, X_speeches_partyid)),
			   ((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'text features + party + debate ID + motion party affiliation'], 'both', (X_speeches_partyidmotion, X_speeches_partyidmotion)),		
			   ((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'debate id + party only'], 'meta', (X_speeches_partyid, X_speeches_partyid)),
			   ((X_speeches, y_speeches), (X_speeches, y_speeches), ['1. 1-step "Speech";', sent_label, 'all meta features only'], 'meta', (X_speeches_partyidmotion, X_speeches_partyidmotion))]

# RUN TWO-STEP MODEL
elif model[0] == '2':
	two_step = True
	# GET FEATURES & LABELS FOR MOTION CLASSIFICATION# 
	for row in debates:
		# get 3 digit debate_id:
		debate_id = str(row[0])
		if len(debate_id) == 1:
			debate_id = '00' + debate_id
		elif len(debate_id) == 2:
			debate_id = '0' + debate_id
		# get text data for motion and speeches and get motion labels:
		motion = debate_id + ' '
		for word in row[2].split()[3:]:
			motion = motion + word + ' '
		if motion not in X_motions:
			X_motions.append(motion)
			y_motions.append(row[3])
	y_motions = np.array(y_motions)
	# DO TRAIN/TEST SPLITS FOR MOTION CLASSIFICATION #
	ids_predictions_list = []
	ids_predictions_dict = {}
	for train_index, test_index in kf.split(X_motions, y_motions):
		motion_ids = [X_motions[i][:3] for i in test_index]
		X_motions_train = [X_motions[i][4:] for i in train_index]
		X_motions_test = [X_motions[i][4:] for i in test_index]
		y_motions_train, y_motions_test = y_motions[train_index], y_motions[test_index]
		motions_train_corpus = vectorizer.fit_transform(X_motions_train)
		motions_test_corpus = vectorizer.transform(X_motions_test) 
		SVM.fit(motions_train_corpus,y_motions_train)
		MLP.fit(motions_train_corpus,y_motions_train)
		SVM_motions_predict = SVM.predict(motions_test_corpus)
		MLP_motions_predict = MLP.predict(motions_test_corpus)
		SVM_motion_acc = SVM_motion_acc+accuracy_score(y_motions_test, SVM_motions_predict)
		SVM_motion_pres = SVM_motion_pres+precision_score(y_motions_test, SVM_motions_predict, pos_label='1')
		SVM_motion_rec = SVM_motion_rec+recall_score(y_motions_test, SVM_motions_predict, pos_label='1')
		SVM_motion_f1 = SVM_motion_f1+f1_score(y_motions_test, SVM_motions_predict, pos_label='1')
		MLP_motion_acc = MLP_motion_acc+accuracy_score(y_motions_test, MLP_motions_predict)
		MLP_motion_pres = MLP_motion_pres+precision_score(y_motions_test, MLP_motions_predict, pos_label='1')
		MLP_motion_rec = MLP_motion_rec+recall_score(y_motions_test, MLP_motions_predict, pos_label='1')
		MLP_motion_f1 = MLP_motion_f1+f1_score(y_motions_test, MLP_motions_predict, pos_label='1')

		ids_preds = zip(motion_ids, MLP_motions_predict)
		for debate in ids_preds:
			ids_predictions_list.append(debate)

	for debate in ids_predictions_list:
		ids_predictions_dict[debate[0]] = debate[1]

	# ADD PREDICTED MOTION POLARITY TO SPEECH UNITS #:
	data.seek(0) # reset csv iterator to zero
	debates_sorted = []
	for row in debates:
		speech = row
		debate_id = str(row[0])
		if len(debate_id) == 1:
			debate_id = '00' + debate_id
		elif len(debate_id) == 2:
			debate_id = '0' + debate_id
		speech.append(ids_predictions_dict[debate_id])
		debates_sorted.append(speech)

	# GET FEATS & SPLIT DATA INTO POS/NEG MOTIONS #
	for row in debates_sorted:
		speech_len = 0
		speech_words = ''
		for utterance in row[6:11]: # utterancess 1-5
			for word in utterance.split():
				speech_len += 1
				speech_words = speech_words + word + ' '
		X_speeches.append(speech_words)
		# debate ID feature:
		deb_id_feat = np.zeros((1,104))[0]
		for debate_id in range(104):
			if int(row[0]) == debate_id:
				deb_id_feat[debate_id] = 1
		# party affiliation feature:
		party_feat = np.zeros((1,len(parties)))[0]
		for party in range(len(parties)):
			if row[13] == parties[party]:
				party_feat[party] = 1
		X_speeches_party.append(party_feat) 
		X_speeches_partyid.append(np.append(party_feat, deb_id_feat))
		# motion party feature:
		motion_party_feat = np.zeros((1,len(motion_parties)))[0]
		for party in range(len(motion_parties)):
			if row[5] == motion_parties[party]:
				motion_party_feat[party] = 1
		X_speeches_partyidmotion.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
		if model == '2a':
			motion_label = 'automatic classification'
			if row[-1] == '1':
				X_speeches1.append(speech_words)
				X_speeches_party_auto1.append(party_feat)
				X_speeches_partyid_auto1.append(np.append(party_feat, deb_id_feat))
				X_speeches_partyidmotion_auto1.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
				if sentiment_label == 'vote':
					sent_label = 'division vote;'	
					y_speeches1.append(int(row[12]))
				else:
					sent_label = 'manually annotated;'
					y_speeches1.append(int(row[11]))
			if row[-1] == '0':
				X_speeches_auto0.append(speech_words)
				X_speeches_party_auto0.append(party_feat)
				X_speeches_partyid_auto0.append(np.append(party_feat, deb_id_feat))
				X_speeches_partyidmotion_auto0.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
				if sentiment_label == 'vote':
					sent_label = 'division vote;'	
					y_speeches0.append(int(row[12]))
				else:
					sent_label = 'manually annotated;'
					y_speeches0.append(int(row[11]))
		elif model == '2b':
			motion_label = 'Govt./opp label'
			if row[4] == '1':
				X_speeches1.append(speech_words)
				X_speeches_party_auto1.append(party_feat)
				X_speeches_partyid_auto1.append(np.append(party_feat, deb_id_feat))
				X_speeches_partyidmotion_auto1.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
				if sentiment_label == 'vote':
					sent_label = 'division vote;'	
					y_speeches1.append(int(row[12]))
				else:
					sent_label = 'manually annotated;'
					y_speeches1.append(int(row[11]))
			if row[4] == '0':
				X_speeches_auto0.append(speech_words)
				X_speeches_party_auto0.append(party_feat)
				X_speeches_partyid_auto0.append(np.append(party_feat, deb_id_feat))
				X_speeches_partyidmotion_auto0.append(np.append(np.append(deb_id_feat, party_feat), motion_party_feat))
				if sentiment_label == 'vote':
					sent_label = 'division vote;'	
					y_speeches0.append(int(row[12]))
				else:
					sent_label = 'manually annotated;'
					y_speeches0.append(int(row[11]))


	y_speeches1 = np.array(y_speeches1)
	y_speeches0 = np.array(y_speeches0)

	data_combos = [((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'text features only'], 'no'),
			   	   ((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'text + party'], 'both', (X_speeches_party_auto1, X_speeches_party_auto0)),
			       ((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'text + debate ID + party'], 'both', (X_speeches_partyid_auto1, X_speeches_partyid_auto0)),
			       ((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'text + debate ID + party + motion'], 'both', (X_speeches_partyidmotion_auto1, X_speeches_partyidmotion_auto0)),
			       ((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'id + party only'], 'meta', (X_speeches_partyid_auto1, X_speeches_partyid_auto0)),
			       ((X_speeches1, y_speeches1), (X_speeches_auto0, y_speeches0), ['2 step "Motion-speech"', motion_label, sent_label, 'all meta only'], 'meta', (X_speeches_partyidmotion_auto1, X_speeches_partyidmotion_auto0))]

combo = data_combos[features]
if two_step == True:
	print('Model:', combo[2][0], '\nSpeech sentiment label:', combo[2][1], '\nMotion sentiment label:', combo[2][2], '\nFeatures:', combo[2][3])
	# PRINT MOTION CLASSIFICATION RESULTS #
	print('\nMotion results:')
	print('    Accuracy (%)  Precision      Recall         F1 score')
	print('SVM', SVM_motion_acc*folds_no, SVM_motion_pres/folds_no, SVM_motion_rec/folds_no, SVM_motion_f1/folds_no)
	print('MLP', MLP_motion_acc*folds_no, MLP_motion_pres/folds_no, MLP_motion_rec/folds_no, MLP_motion_f1/folds_no)
else:
	print('Model:', combo[2][0], '\nSpeech sentiment label:', combo[2][1], '\nFeatures:', combo[2][2])

print('\nSpeech sentiment results:')
SVM_scores = [0,0,0,0]
MLP_scores = [0,0,0,0]
for j in range(2):
	com = combo[j]
	egs_no = len(com[0]) # record number of examples in each split
	for train_index, test_index in kf.split(com[0], com[1]):
		X_speeches_train = [com[0][i] for i in train_index]
		X_speeches_test = [com[0][i] for i in test_index]
		y_speeches_train, y_speeches_test = com[1][train_index], com[1][test_index]
		speeches_train_corpus = vectorizer.fit_transform(X_speeches_train)
		speeches_test_corpus = vectorizer.transform(X_speeches_test)
		# Add meta features:
		if combo[3] == 'both':
			X_speeches_train_meta = [combo[-1][j][i] for i in train_index]
			X_speeches_test_meta = [combo[-1][j][i] for i in test_index]
			speeches_train_corpus = hstack([speeches_train_corpus, X_speeches_train_meta])
			speeches_test_corpus = hstack([speeches_test_corpus, X_speeches_test_meta])
		if combo[3] == 'meta':
			X_speeches_train_meta = [combo[-1][j][i] for i in train_index]
			X_speeches_test_meta = [combo[-1][j][i] for i in test_index]
			speeches_train_corpus = X_speeches_train_meta
			speeches_test_corpus = X_speeches_test_meta
		# fit model:
		SVM.fit(speeches_train_corpus, y_speeches_train)
		MLP.fit(speeches_train_corpus, y_speeches_train)
		# predictions:
		SVM_speeches_predict = SVM.predict(speeches_test_corpus)
		MLP_speeches_predict = MLP.predict(speeches_test_corpus)
		SVM_speech_acc = SVM_speech_acc+accuracy_score(y_speeches_test, SVM_speeches_predict)
		SVM_speech_pres = SVM_speech_pres+precision_score(y_speeches_test, SVM_speeches_predict)
		SVM_speech_rec = SVM_speech_rec+recall_score(y_speeches_test, SVM_speeches_predict)
		SVM_speech_f1 = SVM_speech_f1+f1_score(y_speeches_test, SVM_speeches_predict)
		MLP_speech_acc = MLP_speech_acc+accuracy_score(y_speeches_test, MLP_speeches_predict)
		MLP_speech_pres = MLP_speech_pres+precision_score(y_speeches_test, MLP_speeches_predict)
		MLP_speech_rec = MLP_speech_rec+recall_score(y_speeches_test, MLP_speeches_predict)
		MLP_speech_f1 = MLP_speech_f1+f1_score(y_speeches_test, MLP_speeches_predict)
	SVM_temp = [(SVM_speech_acc*folds_no)*egs_no, (SVM_speech_pres/folds_no)*egs_no, (SVM_speech_rec/folds_no)*egs_no, (SVM_speech_f1/folds_no)*egs_no]
	MLP_temp = [(MLP_speech_acc*folds_no)*egs_no, (MLP_speech_pres/folds_no)*egs_no, (MLP_speech_rec/folds_no)*egs_no, (MLP_speech_f1/folds_no)*egs_no]
	for x in range(len(SVM_scores)):
		SVM_scores[x] += SVM_temp[x]
		MLP_scores[x] += MLP_temp[x]
	SVM_speech_acc, SVM_speech_pres, SVM_speech_rec, SVM_speech_f1 = 0, 0, 0, 0		
	MLP_speech_acc, MLP_speech_pres, MLP_speech_rec, MLP_speech_f1 = 0, 0, 0, 0
if com[0] is X_speeches:
	SVM_scores = [x/2 for x in SVM_scores]
	MLP_scores = [x/2 for x in MLP_scores]

# PRINT SPEECH CLASSIFICATION RESULTS #
print('    Accuracy (%)  Precision      Recall         F1')
print('SVM', SVM_scores[0]/1251, SVM_scores[1]/1251, SVM_scores[2]/1251, SVM_scores[3]/1251)
print('MLP', MLP_scores[0]/1251, MLP_scores[1]/1251, MLP_scores[2]/1251, MLP_scores[3]/1251)
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')

