'''
Created on May 22, 2014

@author: jason
'''
from util import loaddata, plot_roc, roc
from pipeline import Pipeline
from pyearth import Earth
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.base import ClassifierMixin
from grm.grm import GeneralizedRegressor, BinomialLossFunction, LogitLink
from sklearn.svm.classes import SVC

model_dict = {}

class ClassifierPipeline(Pipeline, ClassifierMixin):
    @property
    def classes_(self):
        return self.steps[-1][1].classes_

model_dict['earth_logistic'] = ClassifierPipeline([('earth',Earth(max_degree=1)), ('log',LogisticRegression())])
# model_dict['grm_earth_binomial'] = GeneralizedRegressor(base_regressor=Earth(penalty=0, max_terms=100, thresh=1e-12),
#                              loss_function=BinomialLossFunction(LogitLink()))
# model_dict['grm_earth_binomial_pipeline'] = ClassifierPipeline([('grm',model_dict['grm_earth_binomial']), ('log',LogisticRegression())])
n_estimators = 40
learning_rate = 1.
model_dict['discrete_adaboost'] = AdaBoostClassifier(learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               algorithm="SAMME")
model_dict['real_adaboost'] = AdaBoostClassifier(learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               algorithm="SAMME.R")
model_dict['svc_linear'] = SVC(kernel="linear", C=0.025, probability=True)
model_dict['svc_gamma'] = SVC(gamma=2, C=1, probability=True)
model_dict['real_boost_earth'] = AdaBoostClassifier(base_estimator=ClassifierPipeline([('earth',Earth(max_degree=1)), 
                                                                                  ('log', SVC(probability=True, 
                                                                                             degree=1, kernel='linear'))]),
                                               learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               algorithm="SAMME.R")
 
model_dict['discrete_boost_earth'] = AdaBoostClassifier(base_estimator=ClassifierPipeline([('earth',Earth(max_degree=1)), 
                                                                                  ('log', SVC(probability=True, 
                                                                                             degree=1, kernel='linear'))]),
                                               learning_rate=learning_rate,
                                               n_estimators=n_estimators,
                                               algorithm="SAMME")





