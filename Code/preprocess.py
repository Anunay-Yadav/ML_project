def feature_VADER(Xtext):
    # VADER
    feat_ = []
    sid = SentimentIntensityAnalyzer()
    for tweet in Xtext:
        ss = sid.polarity_scores(tweet)
        f = []
        for i in ss:
            f.append(ss[i])
        feat_.append(f)
    return np.array(feat_)

def polar_wordMPQA(X):
    #mpqa data processing
    dict1_mpqa = {}
    with open("lexicons/2. mpqa.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_mpqa[x[0] + "_" + x[1][:-1]] = 1
    
    #bingluie processing
    dict1_bing = {}
    with open("lexicons/1. BingLiu.csv", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_bing[x[0] + "_" + x[1][:-1]] = 1
    
    #feature extraction
    feat_ = []
    for tokens in X:
        #mpqa
        MPQA = [0,0,0]
        for token in tokens:
            if((token + "_positive") in dict1_mpqa):
                MPQA[0] += 1
            if((token + "_negative") in dict1_mpqa):
                MPQA[1] += 1
            if((token + "_neutral") in dict1_mpqa):
                MPQA[2] += 1
        #bingliu
        BING = [0,0]
        for token in tokens:
            if((token + "_positive") in dict1_bing):
                BING[0] += 1
            if((token + "_negative") in dict1_bing):
                BING[1] += 1
            if((token + "_neutral") in dict1_bing):
                BING[2] += 1
        feat_.append(MPQA + BING)
    return np.array(feat_)

def get_word(word):
    #function to remove all numbers or # in the end
    e = len(word) - 1
    while(e != -1 and (word[e] in "0123456789" or word[e] in "#")):
        e-= 1
    return word[:e + 1]

def aggregate_score_word(X):
    #sentiment140 processing
    dict1_S140 = {}
    with open("lexicons/3. Sentiment140-Lexicon-v0.1/unigrams-pmilexicon.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_S140[x[0] + "_positive"] = float(x[2])
            dict1_S140[x[0] + "_negative"] = float(x[3])
            
    #SENTIWORDNET processing
    dict1_WORDNET= {}
    with open("lexicons/4. SentiWordNet_3.0.0_20130122.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            word = x[4].split(" ")
            word = list(map(get_word,word))
            words = ""
            for i,e in enumerate(word):
                if(i == 0):
                    words += e
                else:
                    words += " " + e
            dict1_WORDNET[words + "_positive"] = float(x[2])
            dict1_WORDNET[words + "_negative"] = float(x[3])
    
    #feature extraction        
    feat_ = []
    for tokens in X:
        sent140 = [0,0]
        wordnet = [0,0]
        afin = [0,0]
        cnt = 0
        for token in tokens:
            if("#" not in token):
                cnt += 1
                #sent140
                if((token + "_positive") in dict1_S140):
                    sent140[0] += dict1_S140[token + "_positive"]
                if((token + "_negative") in dict1_S140):
                    sent140[1] += dict1_S140[token + "_negative"]
                #wordnet
                if((token + "_positive") in dict1_WORDNET):
                    wordnet[0] += dict1_WORDNET[token + "_positive"]
                if((token + "_negative") in dict1_WORDNET):
                    wordnet[1] += dict1_WORDNET[token + "_negative"]
                #afinn
                e = afinn.score(token)
                if(e > 0):
                    afin[0] += e
                else:
                    afin[1] += abs(e)
        res = sent140 + wordnet + afin
        feat_.append([i/cnt for i in res])
    return np.array(feat_)
                    
emotions = set([])
def aggregate_hashtag_scores(X):
    #NRC hashtag processing
    dict1_NRC = {}
    with open("lexicons/7. NRC-Hashtag-Sentiment-Lexicon-v0.1/unigrams-pmilexicon.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_NRC[x[0] + "_positive"] = float(x[2])
            dict1_NRC[x[0] + "_negative"] = float(x[3])
    
    #feature extraction
    feat_ = []
    for tokens in X:
        cnt = 0
        f = [0,0]
        for token in tokens:
            if("#" not in token):
                continue
            cnt += 1
            if((token + "_positive") in dict1_NRC):
                f[0] += dict1_NRC[token + "_positive"]
            if((token + "_negative") in dict1_NRC):
                f[1] += dict1_NRC[token + "_negative"]
        feat_.append([i/(cnt + (i == 0)) for i in f])
    return np.array(feat_)
def emotion_word_count(X):
    #NRC emotion preprocessing
    dict1_NRC = {}
    with open("lexicons/8. NRC-word-emotion-lexicon.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_NRC[x[0] + "_" + x[1]] = int(x[2])
            emotions.add(x[1])
    #feature extraction
    feat_ = []
    for e,tokens in tqdm(enumerate(X)):
        emo_count = [0 for i in range(len(emotions))]
        for token in tokens:
            cnt = 0
            for i in emotions:
                if((token + "_" + i) in dict1_NRC):
                    emo_count[cnt] += dict1_NRC[token + "_" + i]
                cnt += 1
        feat_.append(emo_count)
    return np.array(feat_)

def emotion_word_score(X):
    #NRC emotion preprocessing
    dict1_NRC = {}
    cnt_r = 0
    len1 = 0;
    with open("lexicons/6. NRC-10-expanded.csv", 'r') as fd:
        for line in fd:
            if(cnt_r == 0):
                cnt_r += 1
                continue;
            x = line.split("    ")
            dict1_NRC[x[0]] = [float(i) for i in x[1:]]
            len1 = len(x[1:])
    
    #feature extraction
    feat_ = []
    for e,tokens in tqdm(enumerate(X)):
        emo_score = [0 for i in range(len1)]
        cnt = 0
        for token in tokens:
            if("#" in token):
                continue
            cnt += 1
            if(token in dict1_NRC):
                for i,val in enumerate(dict1_NRC[token]):
                    emo_score[i] += val
        feat_.append([i/cnt for i in emo_score])
    return np.array(feat_)

def emotion_hashtag_score(X):
    #NRC emotion processing
    dict1_NRC = {}
    with open("lexicons/5. NRC-Hashtag-Emotion-Lexicon-v0.2.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            dict1_NRC[x[1] + "_" + x[0]] = float(x[2])

    #feature extraction
    feat_ = []
    for e,tokens in tqdm(enumerate(X)):
        p = 0
        emo_score = [0 for i in range(len(emotions))]
        for token in tokens:
            if("#" not in token):
                continue
            p += 1
            cnt = 0
            for i in emotions:
                if((token + "_" + i) in dict1_NRC):
                    emo_score[cnt] += dict1_NRC[token + "_" + i]
                cnt += 1
        feat_.append([i/(p + (i == 0)) for i in emo_score])
    return np.array(feat_)

def emoticon_score(X):
    #afin emotion processing 
    dict1_afin = {}
    with open("lexicons/9. AFINN-emoticon-8.txt", 'r') as fd:
        for line in fd:
            x = line.split("    ")
            if(float(x[1]) >= 0):
                dict1_afin[x[0] + "_positive"] = (float(x[1]))
                dict1_afin[x[0] + "_negative"] = 0
            else:
                dict1_afin[x[0] + "_positive"] = 0
                dict1_afin[x[0] + "_negative"] = abs(float(x[1]))
                
    #feature extraction
    feat_ = []
    for tokens in X:
        f = [0,0]
        for token in tokens:
            if((token + "_negative") in dict1_afin):
                f[0] += dict1_afin[token + "_negative"]
            if((token + "_positive") in dict1_afin):
                f[1] += dict1_afin[token + "_positive"]
        feat_.append(f)
    return np.array(feat_)
def count_neg_words(X):
    #feature extraction
    feat_ = []
    for tokens in X:
        f = [0]
        p = mark_negation(tokens)
        for i in p:
            if("_NEG" in i):
                f[0] += 1
        feat_.append(f)
    return np.array(feat_)

def word_2_vec(X):
    #word2vec feature extraction
    model = Word2Vec(X, min_count = 0)
    feat_ = []
    for tokens in X:
        f = [-1,-1,-1]
        flag = -1
        
        for token in tokens:
            if(flag == -1):
                flag = 0
                f[0] = model[token]
                f[1] = model[token]
                f[2] = model[token]
            else:
                f[0] = np.add(model[token],f[0]) #average
                f[1] = np.maximum(f[1],model[token]) #maximum
                f[2] = np.minimum(f[2],model[token]) #minimum
        
        if(flag == -1):
            f[0] = model['piss']
            f[1] = model['piss']
            f[2] = model['piss']
            
        f[0] = np.divide(f[0],len(tokens))
        res = []
        for i in f:
            for j in range(i.shape[0]):
                res.append(i[j])
        feat_.append(res)
    return np.array(feat_)

#funciton to process sentences
def preprocess(tweet):
    tweet = word_tokenize(tweet)
    sent = ' '.join([words for words in tweet if words not in cachedStopWords])
    return sent

#unigram construction
def unigram(tweetsarr, tweetTokenized):
    #combining all tokens
    n = len(tweetsarr)
    tokensCombined = []
    for i, tokens in enumerate(tweetTokenized):
        tokensCombined.extend(tokens)

    #get freqdistribution
    analysis = nltk.FreqDist(tokensCombined)

    del tokensCombined
    #mapping construction
    frequencyDict = dict([(m, n) for m, n in analysis.items() if n > 3])
    lenfrequencyDict = len(frequencyDict)
    wordIndex = {}
    for i, key in enumerate(frequencyDict.keys()):
        wordIndex[key] = i
    
    frequencyDict.clear()
#     print(lenfrequencyDict)

    #feature extraction
    featureVector1 = np.zeros([n, lenfrequencyDict])
    for i, tokens in enumerate(tweetTokenized):
        arr = np.zeros([lenfrequencyDict])
        for token in tokens:
            if token in wordIndex:
                arr[wordIndex[token]] = 1
        featureVector1[i] = arr

    return featureVector1

#bigram construction
def bigram(tweetsarr):
    #combining all bigrams
    n = len(tweetsarr)
    bigraminTweet = []
    allbigrams = []
    for tweet in tweetsarr:
        bigrams = []
        for i in range(1,len(tweet)):
            bigrams.append(tweet[i-1] + " " + tweet[i])
        bigraminTweet.append(bigrams)
        allbigrams.extend(bigrams)

    #get freqdistribution
    analysis = nltk.FreqDist(allbigrams)
    del allbigrams
    frequencybigramDict = dict([(m, n) for m, n in analysis.items() if n > 5])
    lenfrequencybigramDict = len(frequencybigramDict)

    bigramIndexDict = {}
    for i, key in enumerate(frequencybigramDict.keys()):
        bigramIndexDict[key] = i
    frequencybigramDict.clear()

    #feature extraction
    featureVector2 = np.zeros([n, lenfrequencybigramDict])
    for i, bigramTweet in enumerate(bigraminTweet):
        arr = np.zeros([lenfrequencybigramDict])
        for bigram in bigramTweet:
            if bigram in bigramIndexDict:
                arr[bigramIndexDict[bigram]] = 1
        featureVector2[i] = arr
    del bigraminTweet

    return featureVector2


def feature_extraction(Xtext,X):
    VADER = feature_VADER(Xtext) #1
    polar = polar_wordMPQA(X) #2
    agg_word_score = aggregate_score_word(X) #3
    agg_hash_score = aggregate_hashtag_scores(X) #4
    emot_word = emotion_word_count(X) #5
    emot_word_score = emotion_word_score(X) #6
    emote_hash_score = emotion_hashtag_score(X) #7
    emoticon = emoticon_score(X)#8
    neg_words = count_neg_words(X)#9
    uni = unigram(Xtext,X) #unigram feature extraction
    bi = bigram(X) #bigram feature extraction
    word_vec = word_2_vec(X)#10
    vectorizer = TfidfVectorizer() #11 tfidf
    word_feat = vectorizer.fit_transform(Xtext).toarray()
    
    a = len(Xtext)
#     b = word_vec.shape[1] + word_feat.shape[1] + uni.shape[1] + bi.shape[1] + VADER.shape[1] + polar.shape[1] + agg_word_score.shape[1] + agg_hash_score.shape[1] + emot_word.shape[1] + emot_word_score.shape[1] + emote_hash_score.shape[1] + emoticon.shape[1] + neg_words.shape[1]
    b = uni.shape[1] + bi.shape[1] + VADER.shape[1] + polar.shape[1] + agg_word_score.shape[1] + agg_hash_score.shape[1] + emot_word.shape[1] + emot_word_score.shape[1] + emote_hash_score.shape[1] + emoticon.shape[1] + neg_words.shape[1]
  
     #feature combination
    feature = np.zeros([a, b])
    for i in range(len(Xtext)):
        feature[i] = np.concatenate((bi[i],uni[i],VADER[i],polar[i],agg_word_score[i],agg_hash_score[i],emot_word[i],emot_word_score[i],emote_hash_score[i],emoticon[i],neg_words[i]))
    print(feature.shape)
    
    return feature

