'''
Created on May 13, 2014

@author: jason
'''
from util import loaddata, plot_roc, roc
from pipeline import Pipeline
from pyearth.earth import Earth
from sklearn.linear_model.logistic import LogisticRegression
from matplotlib import pyplot
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.base import ClassifierMixin
class ClassifierPipeline(Pipeline, ClassifierMixin):
    @property
    def classes_(self):
        return self.steps[-1][1].classes_
wtrain, wtest, Xtrain, Xtest, ytrain, ytest = loaddata(nrows=5000)
classifier = ClassifierPipeline([('earth',Earth(max_degree=1)), ('log',LogisticRegression())])
model = classifier#AdaBoostClassifier(base_estimator=classifier)
model = model.fit(Xtrain, ytrain)
ptest = model.predict_proba(Xtest)[:,1]

print model.steps[0][1].trace()
print model.steps[0][1].summary()
pyplot.figure()
plot_roc(*roc(ytest, ptest))
pyplot.show()
# print train[train.columns[-5:-3]]



