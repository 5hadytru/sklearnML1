import sklearn
import numpy
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import metrics

digits = load_digits()
# Scales digits down to values between -1 and 1 so they're easier to work with since they're big
data = scale(digits.data)
y = digits.target

k = 10
# Returns number of instances and number of attributes
samples, features = data.shape

# Takes classifier, name, and data as arguments and scores classifier with numerous scorers
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

'''
To interpret scores: https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
n_clusters = number of clusters/centroids
init = random (randomly place initial centroids), kmeans++ (places them at equal distances from
each other on the grid), or ndarray (allows you to code initial centroids)
n_init = number of times it runs randomly seeded centroids before the best seed is chosen to use
Numerous other parameters exist
'''
clf = KMeans(n_clusters=k, init='random', n_init=10)
bench_k_means(clf, 'AYO', data)
