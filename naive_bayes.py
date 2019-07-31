import pandas as pd
import numpy as np
import math

class Counts:
    def __init__(self):
        # Training set
        self.trg_word_counts = {}
        self.trg_class_word_counts = {}
        self.trg_num_words_in_class = {}
        self.trg_class_occurrences = {}
        self.trg_abstract_sets = []
        self.idf_counts = {}
        
        # Test set
        self.tst_frequencies = []
        
c = Counts()
stop_words = {"a", "about", "above", "after", "again", "against", "all", "am", "an", "and", 
              "any", "are", "aren't", "as", "at", "be", "because", "been", "before", 
              "being", "below", "between", "both", "but", "by", "can't", "cannot", "could", 
              "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", 
              "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", 
              "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", 
              "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", 
              "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", 
              "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", 
              "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
              "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", 
              "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", 
              "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", 
              "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", 
              "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", 
              "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", 
              "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", 
              "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", 
              "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves"}

def classify(trg, tst, ext2 = True, ext1 = True):
    c.__init__()
    return multinomial_naive_bayes(trg, tst, ext1, ext2)
    
# --- Preprocessing ---
def process(abstracts, ext2):
    processed_abstracts = []
    for abstract in abstracts:
        abstract_list = abstract.split()
        abstract_list = apply_word_filters(abstract_list, ext2)
        processed_abstracts.append(abstract_list)

    return processed_abstracts

def apply_word_filters(abstract_list, ext2):
    # Extension 2 Part 1: Filtering out stop words
    if ext2:
        abstract_list = list(filter(lambda word: word not in stop_words, abstract_list))
    return abstract_list

# -------------------------------
# --- Multinomial Naive Bayes ---
def get_counts(trg, tst):
    # Get word counts by considering all documents in trg as one
    for i in range(len(trg)):
        class_ = trg["class"][i]
        increment_dict(c.trg_class_occurrences, class_)
        c.trg_abstract_sets.append(set())
        for word in trg["abstract"][i]:            
            if class_ not in c.trg_class_word_counts:
                c.trg_class_word_counts[class_] = {}
            if class_ not in c.trg_num_words_in_class:
                c.trg_num_words_in_class[class_] = 0
                
            increment_dict(c.trg_word_counts, word)
            increment_dict(c.trg_class_word_counts[class_], word)
            increment_dict(c.trg_num_words_in_class, class_)
            c.idf_counts[word] = 1
            

    # Get word frequencies for each individual document in tst
    for i in range(len(tst)):
        c.tst_frequencies.append(dict())
        for word in tst["abstract"][i]:
            increment_dict(c.tst_frequencies[i], word)

def increment_dict(dict_, key):
    if key in dict_:
        dict_[key] += 1
    else:
        dict_[key] = 1

def get_idf(word, trg):
    # Extension 2 Part 2: Using the Inverse Document Frequency (IDF)
    numerator = len(trg)
    if word in c.idf_counts:
        denominator = c.idf_counts[word]
    else:
        denominator = 1

    return math.log10(len(trg) / denominator)

def multinomial_naive_bayes(trg, tst, ext1, ext2):
    print("Processing the training abstracts...")
    trg["abstract"] = process(trg["abstract"], ext2)
    
    print("Processing the test abstracts...")
    tst["abstract"] = process(tst["abstract"], ext2)

    print("Training the MNBC...")
    get_counts(trg, tst)

    print("Classifying the test abstracts...")
    classes = set(np.unique(trg["class"]))
    class_probabilities = {}
    prior = {}
    log = math.log10
    for class_ in classes:
        prior[class_] = c.trg_class_occurrences[class_]/len(trg)

    alpha = 1
    labels = []
        
    for i in range(len(tst)):
        class_probabilities = prior.copy()
        for word in c.tst_frequencies[i]:
            idf = get_idf(word, trg)
            for class_ in classes:
                frequency = c.tst_frequencies[i][word]
                if ext2:
                    frequency *= idf
                if word in c.trg_class_word_counts[class_]:
                    numerator = c.trg_class_word_counts[class_][word] + alpha
                else:
                    numerator = alpha
                denominator = c.trg_num_words_in_class[class_] + len(c.trg_word_counts)
                
                # Extension 1: Incorporating the Complement Naive Bayes (CNB) formula
                if ext1:
                    comp_numerator = alpha
                    comp_denominator = len(c.trg_word_counts)
                    for cl in classes.difference({class_}):
                        if word in c.trg_class_word_counts[cl]:
                            comp_numerator += c.trg_class_word_counts[cl][word] + alpha
                        comp_denominator += c.trg_num_words_in_class[cl]
                    class_probabilities[class_] -= frequency * log(comp_numerator/comp_denominator)
                                
                class_probabilities[class_] += frequency * log(numerator/denominator)
                

        label = max(class_probabilities, key = class_probabilities.get)
        labels.append(label)

    return labels

# ---------------------------------------------
# --- Testing with 10-fold cross-validation ---
def cross_validate(num_folds = 10, ext1 = True, ext2 = True):
    print("ext1 = {}, ext2 = {}".format(ext1, ext2))
    # Split the training set
    trg_ = pd.read_csv("trg.csv")
    trgs = []
    num_folds = num_folds
    accuracies = []
    for i in range(1, num_folds + 1):
        start = (i-1)*int((len(trg_)/num_folds))
        stop = i*int((len(trg_)/num_folds))
        trgs.append(trg_[start:stop])

    # Run cross-validation
    for i in range(num_folds):
        print("Fold {} of {}".format(i+1, num_folds))
        trg = pd.concat(trgs[:i] + trgs[i+1:])
        tst_answers = trgs[i]
        trg.index = range(len(trg))
        tst_answers.index = range(len(tst_answers))
        tst = tst_answers.drop(columns = ["class"])

        # Get accuracy:
        tst_class_predictions = classify(trg, tst, ext1, ext2)
        tst["class"] = tst_class_predictions
        correct = 0
        total = len(tst)

        for i in range(len(tst)):
            if list(tst["class"])[i] == list(tst_answers["class"])[i]:
                correct += 1

        accuracy = 100 * correct/total
        accuracies.append(accuracy)
        print()

    print("Accuracies from all {} folds with ext1 = {}, ext2 = {}".format(
        num_folds, ext1, ext2))
    print([round(accuracy, 2) for accuracy in accuracies])
    print("Mean accuracy after cross-validation: " + str(sum(accuracies)/len(accuracies)) + "%")
    print()
    
# -------------------------------------------------
# --- Write out the final output to a .csv file ---
def write_output(ext1 = True, ext2 = True):
    trg = pd.read_csv("trg.csv")
    tst = pd.read_csv("tst.csv")
    tst["class"] = classify(trg, tst, ext1, ext2)
    tst.drop(["abstract"], axis = 1).to_csv("tst_kaggle.csv", index = False)

if __name__ == "__main__":
    # Extension 1: Incorporating the Complement Naive Bayes formula -- set with the ext1 flag
    # Extension 2: Filtering out stop words and using the IDF -- set with the ext2 flag
    folds = 10
##    cross_validate(num_folds = folds, ext1 = False, ext2 = False)
##    cross_validate(num_folds = folds, ext1 = True, ext2 = False)
##    cross_validate(num_folds = folds, ext1 = False, ext2 = True)
##    cross_validate(num_folds = folds, ext1 = True, ext2 = True)
    
    write_output(ext1 = True, ext2 = True)

    
