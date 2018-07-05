import sys
from lexsub_xml import read_lexsub_xml
# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
import gensim
import numpy as np

# Participate in the 4705 lexical substitution competition (optional): Yes
# Alias: ah3262

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    syns = wn.synsets(lemma,pos)
    for sys in syns:
        for lem in sys.lemmas():
            if lem.name() not in possible_synonyms and lem.name() != lemma:
                nice = lem.name()
                nice = nice.replace('-',' ')
                nice = nice.replace('_',' ')
                possible_synonyms.append(nice)
    print(len(possible_synonyms))
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    lemma = context.lemma
    pos = context.pos
    possible_synonyms = []
    counts = []
    syns = wn.synsets(lemma,pos)
    for sys in syns:
        for lem in sys.lemmas():
            if lem.name() != lemma:
                nice = lem.name()
                nice = nice.replace('-',' ')
                nice = nice.replace('_',' ')
                if nice in possible_synonyms:
                    index = possible_synonyms.index(nice)
                    counts[index] = counts[index] + lem.count()
                else:
                    possible_synonyms.append(nice)
                    counts.append(lem.count())
    print(possible_synonyms)
    highest = max(counts)
    index = counts.index(highest)
    return possible_synonyms[index] # replace for part 2

def wn_simple_lesk_predictor(context):
    lemma = context.lemma
    pos = context.pos
    left = context.left_context
    right = context.right_context
    possible_synonyms = []
    counts = []
    context = []
    for word in left:
        word = word.lower()
        if word not in context:
            context.append(word)
    for word in right:
        word = word.lower()
        if word not in context:
            context.append(word)   
    syns = wn.synsets(lemma,pos)
    for syn in syns:
        tokens = []
        defs = syn.definition()
        defs = defs.split()
        for word in defs:
            word = word.lower()
            if word not in tokens:
                tokens.append(word)
        hypernyms = syn.hypernyms()
        for hypernym in hypernyms:
            hypernym = hypernym.name()
            hypernym = hypernym.split('.')[0]
            h_syns = wn.synsets(hypernym)
            for h_syn in h_syns:
                h_defs = h_syn.definition()
                h_defs = h_defs.replace('-',' ')
                h_defs = h_defs.replace('_',' ')
                h_defs = h_defs.split()
                for word in h_defs:
                    word = word.lower()
                    if word not in tokens:
                        tokens.append(word)
                for example in h_syn.examples():
                    example = example.replace('-',' ')
                    example = example.replace('_',' ')
                    example = example.split()
                    for word in example:
                        word = word.lower()
                        if word not in tokens:
                            tokens.append(word)
        for example in syn.examples():
            example = example.replace('-',' ')
            example = example.replace('_',' ')
            example = example.split()
            for word in example:
                word = word.lower()
                if word not in tokens:
                    tokens.append(word)
        stop_words = stopwords.words('english') 
        for lem in syn.lemmas():
            if lem.name() != lemma:
                tokens = [word for word in tokens if word not in stop_words]
                nice = lem.name()
                nice = nice.replace('-',' ')
                nice = nice.replace('_',' ')
                if nice in possible_synonyms:
                    index = possible_synonyms.index(nice)
                    counts[index] = counts[index] + lem.count()
                else:
                    possible_synonyms.append(nice)
                    counts.append(lem.count() + 100*len(set(tokens).intersection(context)))
    highest = max(counts)
    index = counts.index(highest)
    return possible_synonyms[index]            

class Word2VecSubst(object):
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)
    def predict_nearest(self,context):
        possible_synonyms = []
        scores = []
        lemma , pos = context.lemma, context.pos
        syns = wn.synsets(lemma,pos)
        for sys in syns:
            for lem in sys.lemmas():
                if lem.name() not in possible_synonyms and lem.name() != lemma:
                    nice = lem.name()
                    nice = nice.replace('-',' ')
                    nice = nice.replace('_',' ')
                    possible_synonyms.append(nice)
        model = self.model
        for candidate in possible_synonyms:
            try:
                scores.append(model.similarity(candidate,lemma))
            except KeyError:
                print("not in vocabulary")
                scores.append(0.0)
        highest = max(scores)
        index = scores.index(highest)
        return possible_synonyms[index]
    def predict_nearest_with_context(self, context):
        possible_synonyms = []
        scores = []
        lemma , pos = context.lemma, context.pos
        syns = wn.synsets(lemma,pos)
        for sys in syns:
            for lem in sys.lemmas():
                if lem.name() not in possible_synonyms and lem.name() != lemma:
                    nice = lem.name()
                    nice = nice.replace('-',' ')
                    nice = nice.replace('_',' ')
                    possible_synonyms.append(nice)
        model = self.model
        left = context.left_context[::-1]
        right = context.right_context
        stop_words = stopwords.words('english')
        left = [word for word in left if word not in stop_words]
        right = [word for word in right if word not in stop_words]
        context = []
        scope = 5
        i = 0
        while i<scope and i<len(left):
            word = left[i]
            word = word.lower()
            if word not in context:
                context.append(word)
            i += 1
        i = 0
        while i<scope and i<len(right):
            word = right[i]
            word = word.lower()
            if word not in context:
                context.append(word)
            i += 1
        vector = np.copy(model.wv[lemma])
        for word in context:
            try:
                vector += np.copy(model.wv[word])
            except KeyError:
                vector = vector
        for candidate in possible_synonyms:
            try:
                v1, v2 = model.wv[candidate], vector
                cos = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                scores.append(cos)
            except KeyError:
                print("not in vocabulary")
                scores.append(0.0)
        highest = max(scores)
        index = scores.index(highest)
        return possible_synonyms[index]

    def predict_best(self, context):        
        possible_synonyms = []
        scores = []
        lemma , pos = context.lemma, context.pos
        syns = wn.synsets(lemma,pos)
        for sys in syns:
            for lem in sys.lemmas():
                if lem.name() not in possible_synonyms and lem.name() != lemma:
                    nice = lem.name()
                    nice = nice.replace('-',' ')
                    nice = nice.replace('_',' ')
                    possible_synonyms.append(nice)
        model = self.model
        left = context.left_context[::-1]
        right = context.right_context
        stop_words = stopwords.words('english')
        left = [word for word in left if word not in stop_words and len(word)>1]
        right = [word for word in right if word not in stop_words and len(word)>1]
        context = []
        scope1 = 2
        scope2 = 4
        i = 0
        while i<scope2 and i<len(left):
            word = left[i]
            word = word.lower()
            if word not in context:
                context.append(word)
            i += 1
        i = 0
        while i<scope1 and i<len(right):
            word = right[i]
            word = word.lower()
            if word not in context:
                context.append(word)
            i += 1
        vector = np.copy(model.wv[lemma])
        for word in context:
            try:
                vector += np.copy(model.wv[word])
            except KeyError:
                vector = vector
        for candidate in possible_synonyms:
            try: 
                vector0 = np.copy(model.wv[candidate])
                for word in context:
                    try:
                        vector0 += np.copy(model.wv[word])
                    except KeyError:
                        vector0 = vector0
                v1, v2 = vector0, vector
                cos = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
                scores.append(cos)
            except KeyError:
                print("not in vocabulary")
                scores.append(0.0)
        highest = max(scores)
        index = scores.index(highest)
        return possible_synonyms[index]

if __name__=="__main__":
    # At submission time, this program should run your best predictor (part 6).
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)
    #print(get_candidates('slow','a'))
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        prediction = predictor.predict_best(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
