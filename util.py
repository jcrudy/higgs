'''
Created on May 13, 2014

@author: jason
'''
import pandas
from config import train_file, test_file
from pandas.core.array import NA
import numpy
from sklearn.cross_validation import train_test_split
from sklearn.metrics.metrics import roc_curve, auc
from matplotlib import pyplot
from sklearn.base import clone
import os
import pickle
from kagglecode import AMS
import scipy.stats

ylabel = 'Label'
weightlabel = 'Weight'
idlabel = 'EventId'
train_size = .5
test_size = .5

def naify(dataframe):
    for col in dataframe.columns:
        if numpy.any(dataframe[col] == -999):
            dataframe[col + '_NA'] = dataframe[col] == -999
    return dataframe

def loadtrain(nrows=None):
    return naify(pandas.read_csv(train_file, nrows=nrows))

def loadtest(nrows=None):
    return naify(pandas.read_csv(test_file, nrows=nrows))

def extractx(train):
    X = train.copy()
    try:
        del X[ylabel]
    except KeyError:
        pass
    try:
        del X[weightlabel]
    except KeyError:
        pass
    try:
        del X[idlabel]
    except KeyError:
        pass
    return X

def extracty(train):
    y = train[ylabel] == 's'
    return y

def extractweight(train):
    w = train[weightlabel]
    return w

def extractid(train):
    ids = train[idlabel]
    return ids

def splitwxy(train):
    w = extractweight(train)
    X = extractx(train)
    y = extracty(train)
    ids = extractid(train)
    return w, X, y, ids

def splittraintest(w, X, y, ids, train_size=train_size, test_size=test_size):
    w_train, w_test, X_train, X_test, y_train, y_test, id_train, id_test = \
        train_test_split(w, X, y, ids, train_size=train_size, test_size=test_size)
    return w_train, w_test, X_train, X_test, y_train, y_test, id_train, id_test

def loaddata(nrows=None):
    return splittraintest(*splitwxy(loadtrain(nrows=nrows)))

def load_submission_data(nrows=None):
    data = loadtest(nrows)
    X = extractx(data)
    ids = extractid(data)
    return X, ids

def fit_and_validate(base_model, wtrain, wtest, Xtrain, Xtest, ytrain, ytest):
    model = clone(base_model)
    
    model = model.fit(Xtrain, ytrain)
    try:
        ptrain = model.predict_proba(Xtrain)[:,1]
        ptest = model.predict_proba(Xtest)[:,1]
    except AttributeError:
        ptrain = model.predict_proba(Xtrain)
        ptest = model.predict(Xtest)
    
    auc_train, _, _ = roc(ytrain, ptrain)
    auc_test, fpr, tpr = roc(ytest, ptest)
    cutoff, ams_train = optimize_cutoff(ytrain, ptrain, wtrain)
    ams_test = compute_ams(ytest, ptest, wtest, cutoff)
    
    return model, auc_train, auc_test, fpr, tpr, ams_train, ams_test, cutoff

def generate_submission(model, X, cutoff, ids):
    try:
        p = model.predict_proba(X)[:,1]
    except AttributeError:
        p = model.predict(X)
    y_bool = p >= cutoff
    y = numpy.empty(shape=y_bool.shape, dtype=object)
    y[y_bool] = 's'
    y[~y_bool] = 'b'
    result = pandas.DataFrame(ids, columns=['EventId'])
#     result['EventId'] = ids
    result['RankOrder'] = scipy.stats.rankdata(p, method='ordinal').astype(int)
    result['Class'] = y
    return result

def run_experiment(model_dict, output_dir, nrows=None):
    wtrain, wtest, Xtrain, Xtest, ytrain, ytest, id_train, id_test = loaddata(nrows=nrows)
    Xsub, idsub = load_submission_data()
    data = []
    for name, base_model in model_dict.iteritems():
        model, auc_train, auc_test, fpr, tpr, ams_train, \
            ams_test, cutoff = fit_and_validate(base_model, wtrain, wtest, Xtrain, Xtest, ytrain, ytest)
        pyplot.figure()
        plot_roc(auc_test, fpr, tpr)
        pyplot.savefig(os.path.join(output_dir, name + '.png'))
        with open(os.path.join(output_dir, name + '.pickle'), 'w') as outfile:
            pickle.dump(model, outfile)
        data.append([name, auc_train, auc_test, ams_train, ams_test, cutoff])
        submission = generate_submission(model, Xsub, cutoff, idsub)
        submission.to_csv(os.path.join(output_dir, name + '_submission.csv'), index=False)
    dataframe = pandas.DataFrame(data, columns=['model','auc_train','auc_test','ams_train','ams_test','cutoff'])
    dataframe.to_csv(os.path.join(output_dir, 'results.csv'), index=False)

def compute_ams(y, p, w, cutoff):
    y_hat = p >= cutoff
    b = numpy.sum(w[y_hat & (~(y==1))])
    s = numpy.sum(w[y_hat & (y==1)])
    ams = AMS(s,b)
    return ams

def optimize_cutoff(y, p_hat, w):
    # Order by decreasing p_hat
    order = numpy.argsort(p_hat)[::-1]
    best_cutoff = None
    best_ams = None
    for idx in order:
        cutoff = p_hat[idx]
        ams = compute_ams(y, p_hat, w, cutoff)
        if best_ams is None or ams > best_ams:
            best_cutoff = cutoff
            best_ams = ams
    return best_cutoff, best_ams

    
def roc(y, p_hat):
    fpr, tpr, thresholds = roc_curve(y, p_hat)
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, fpr, tpr

def plot_roc(roc_auc, fpr, tpr):
    pyplot.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pyplot.plot([0, 1], [0, 1], 'k--')
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.0])
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('Receiver operating characteristic example')
    pyplot.legend(loc="lower right")
    
    


