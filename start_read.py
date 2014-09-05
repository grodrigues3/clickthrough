
import csv, time
from collections import defaultdict
import scipy.sparse as ssp
import numpy as np
import random as rnd
import pdb
import math
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, LogisticRegression
import cPickle, traceback
from sklearn.grid_search import GridSearchCV
#upper= 10**7
#sampleIndices = set(rnd.sample(range(upper), breakVal))

numCats = 26
numTraining = 45840618 
def count_categories(filename, itemsLimit):

    with open(filename, 'r') as f:
        myReader = csv.DictReader(f, delimiter=",", quotechar='"')
        s = time.time()
        labels = []
        c_counts = [defaultdict(lambda:[0,0]) for i in range(numCats)]
        """
        This part reads the data file and creates the c_counts list
        c_counts contains all values within a CXX category as well as 
        how often the value appears.

        Important Data Structure: c_counts
        Suppose the category: shoes appears for C8 100 times and
        80 times results in a label of one
        then c_counts[7]["shoe"] = [100, 80]
        """
        for row, values in enumerate(myReader):
            for i in range(1,numCats+1): #C1-C26
                cVal = "C"+str(i)
                c_counts[i-1][values[cVal]][0]+=1
                if int(values['Label']):
                    c_counts[i-1][values[cVal]][1]+=1
            if row >= itemsLimit-1:
                break
    print "DONE"
    print "Elapsed Time", time.time() - s
    return c_counts
    
    
def print_conditional_probs(c_counts, labels):
    printLimit = 100
    for i in range(1,27):
        currentDict = c_counts[i-1]
        cVal = "C"+str(i)
        print cVal, len(currentDict)
        if len(currentDict) < printLimit:
            print 'This is for C'+str(i-1)
            print "Number of Entries", len(currentDict)
            for val in currentDict:
                print val, currentDict[val], currentDict[val][1] / float(currentDict[val][0])
    print labels


def build_index_dict(c_counts):
    #make this a small number like .00005 to cut down total num of  cats
    trimPercent = 0 
    numCats = len(c_counts)
    matDict = [{} for i in range(numCats)]
    catCount = 0
    for CXX, cat in enumerate(c_counts):
        for val in cat:
           matDict[CXX][val] = catCount
           catCount+=1
    print catCount-1
    return matDict,catCount


def build_full_matrix(filename, indexDict, totalCats, itemsLimit=None, model=None): #,offset=False):
    """FIRST FORM THE CATEGORICAL PART"""
    if not itemsLimit:
        itemsLimit = 10**5
    else:
        samples = rnd.sample(xrange(numTraining), itemsLimit)
    """
    with open(filename, 'r') as g:
        numRows = -1 #to account for the headerRow
        for line in g:
            numRows +=1
        itemsLimit = numRows
    """
    notFound = 0
    with open(filename, 'r') as f, open('myOut.csv', 'a') as g:
        myReader = csv.DictReader(f, delimiter=",",  quotechar='"')
        #print myReader.fieldnames
        acceptable =  ["I"+str(i) for i in range(1,14)]
        firstHalf = np.zeros((itemsLimit, 13), dtype=np.int)
        secondHalf = ssp.lil_matrix((itemsLimit, totalCats))
        ids = []
        labels = []
        row = 0
        count = 0
        for values in myReader:
            """
            We form the matrices in two parts:
                1) The encoded categorical features
                2) The count features which are already numerical
            """
            if "Label" in values:
                labels += [int(values['Label'])]
            ids += [values["Id"]]
            for i in range(1,numCats+1): #C1-C26
                cVal = "C"+str(i)
                try:
                    col = indexDict[i-1][values[cVal]]
                    secondHalf[row, col] = 1
                except:
                    #print "The value for this column is not encoded"
                    notFound +=1
            for ind, feat in enumerate(acceptable):
                try:
                    firstHalf[row, ind] = int(values[feat]) if values[feat] else 0
                except:
                    print "There was an error"
                    traceback.print_exc()
                    print feat, values[feat]
            if row%1000 == 0:
                print "Number of Rows Processed", count*itemsLimit+row+1
            if model and (row+1)%itemsLimit==0 and row>0:
                newMat = ssp.hstack([firstHalf, secondHalf])
                probs = model.predict_proba(newMat)
                print "Writing to File"
                for id, prob in zip(ids, probs):
                        g.write(str(id) + "," + str(prob[1]) + "\n")
                #Resetting all the values so I don't run out of memory 
                firstHalf = np.zeros((itemsLimit, 13), dtype=np.int)
                secondHalf = ssp.lil_matrix((itemsLimit, totalCats))
                row = 0
                count +=1
                ids = []
            if row >= itemsLimit -1 and not model:
                print "breaking"
                print filename
                break
            row +=1
    print "The total number of cells not used: ", notFound
    newMat = ssp.hstack([firstHalf, secondHalf])
    if model:
        probs = model.predict_proba(newMat)
        #probs = model.predict_proba(firstHalf)
        with open('myOut.csv', 'a') as f:
            print len(ids)
            print probs.shape
            for id, prob in zip(ids, probs):
                    f.write(str(id) + "," + str(prob[1]) + "\n")
        return
    print newMat.shape
    return ids, newMat, labels
            
def train_classifier(data, labels):
    
    nIter = 50
    alphaVals = [10**i for i in range(3,5)]
    params = { "loss": ["log"],
        "penalty": ['l1', 'l2'],
        "n_iter": [nIter],
        "alpha": alphaVals
    }
    params_log = { 
        "penalty": ['l2'] ,
        "C": [10**i for i in range(-3,-1)]
    }
    #sgd = SGDClassifier()
    sgd = LogisticRegression()
    clf = GridSearchCV(sgd, params_log)
    #data = data.tocsr()[:, 0:13]
    train, val, t_labs, val_labs = train_test_split(data,labels, train_size=.2, random_state=44)
    s = time.time()
    clf.fit(train, t_labs)
    print "Elapsed Training Time for ", len(params_log['C']), 'regularization vals: ', time.time() - s
    print clf.best_params_ 
    

    print "The Validation Score: ", clf.score(val, val_labs)
    probs =  clf.predict_proba(val)
    print "The log loss for the validation set is"
    print log_loss(probs[:,1], val_labs)
    return clf


def log_loss(probs, labels):
    n = len(probs)
    loss = 0
    for pred, true in zip(probs, labels):
        if true:
            if pred == 0:
                pred = 1e-7
            loss += true*math.log(pred) 
        else:
            if pred == 1:
                pred = 1 - 1e-7
            loss += (1-true) * math.log(1-pred) 
    loss = -1/float(n) * loss
    return loss

def test_model(model, filename, matDict, catCount):
    with open('myOut.csv', 'w') as f:
        f.write("Id,Predicted\n")
    #ids, data, labels = build_full_matrix(filename, matDict, catCount, None, True)
    build_full_matrix(filename, matDict, catCount, None, model)
    #probs = model.predict_proba(data)



if __name__ == "__main__":
    limit = 2*10**6
    #c_counts = count_categories("train.csv", limit)
    #matDict, catCount = build_index_dict(c_counts)
    #ids, data, labels = build_full_matrix('train.csv',matDict, catCount, limit)
    matDict, catCount = cPickle.load(open('indexDict.pkl', 'r'))
    #joblib.dump([ids,data,labels], 'firstmil_train.pkl')
    ids, data,labels = joblib.load('first100_train.pkl')
    #print "Done loading data"
    model = train_classifier(data, labels)
    del data, labels, ids
    start = time.time()
    test_model(model, 'test.csv', matDict, catCount)
    print "Elapsed Time for Inference Only: ", time.time() - start
