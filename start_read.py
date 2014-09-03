
import csv, time
from collections import defaultdict
import scipy.sparse as ssp
import numpy as np
import random as rnd
import pdb
from sklearn.externals import joblib

numCats = 26
upper= 10**7
breakVal = 10**6
sampleIndices = set(rnd.sample(range(upper), breakVal))


def read_file():
    with open('train.csv', 'r') as f:
        myReader = csv.DictReader(f, delimiter=",", quotechar='"')
        #print f.readline()
        s = time.time()
        featNames = myReader.fieldnames
        labels = defaultdict(lambda:0)
        c_counts = [defaultdict(lambda:[0,0]) for i in range(numCats)]
        for num, row in enumerate(myReader):
            for i in range(1,numCats+1): #C1-C26
                cVal = "C"+str(i)
                c_counts[i-1][row[cVal]][0]+=1
                if int(row['Label']):
                    c_counts[i-1][row[cVal]][1]+=1
            labels[row['Label']] +=1
            if num >= breakVal:
                break
    pdb.set_trace()
    joblib.dump([c_counts], 'category_dict.pkl')
    print "DONE"
    print "Elapsed Time", time.time() - s

    return labels, c_counts
    
    
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


def create_sparse(c_counts=None, filename=None):
    if not c_counts:
        c_counts = joblib.load('category_dict.pkl')
    matDict = {}
    s = 0
    for cat in c_counts:
        for val in cat:
            if val not in matDict:
               matDict[val] = s
               s+=1
    print len(matDict), 'Total Number of Cats'
    
if __name__ == "__main__":
    labs, c_counts = read_file()
    create_sparse(c_counts, "train.csv")
    #print_conditional_probs(c_counts, labs)

