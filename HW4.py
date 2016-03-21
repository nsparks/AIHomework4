from random import shuffle
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
  # get data from csv files
  inputs_15, targets = load_data("./2015.csv")
  inputs_16, targets_16 = load_data("./2016.csv")
  k_inputs_15, targets = load_data("./2015.csv", linear=False)
  k_inputs_16, k_targets_16 = load_data("./2016.csv", linear=False)
  
  #
  # Linear Regressions
  #
  lr = linear_regression(alpha=1E-7, decay_rate=10000)
  
  # train regression
  output_15 = lr.train_regression(inputs_15, targets, max_loops=50000)
  
  # predicts outputs
  predictions_16 = lr.predict_regression(inputs_16)
  
  # plot 1: learning curve
  fig1 = plt.figure()
  plt.subplot(2,1,1)
  plt.plot(range(len(output_15)), output_15)
  plt.xlabel("Iterations")
  plt.ylabel("SSE (Degrees)")
  plt.title("Error vs. Training iteration")
  
  # plot 2: Predictions vs Actual Values
  plt.subplot(2,1,2)
  plt.plot(range(len(targets_16)), predictions_16, label="Predicted Values")
  plt.subplot(2,1,2)
  plt.plot(range(len(targets_16)), targets_16, label="Target Values")
  plt.legend(loc=4)
  plt.xlabel("Days")
  plt.ylabel("Temperature (Degrees)")
  plt.title("Linear Regression: 2016 Output temps vs. expected temps")
  
  #
  # K-Means Clusters
  #
  
  k = k_means_cluster()
  k.cluster(k_inputs_15)
  predictions = k.predict(k_inputs_16)
  
  errors = []
  for i in range(2,8):
    e = 0
    ki = k_means_cluster(i)
    
    # cluster multiple times for each k and average the SSE
    loops = 20
    for j in range(loops):
      ki.cluster(k_inputs_15)
      e += ki.sse(k_inputs_16, k_targets_16)
    e /= loops
    errors.append(e)
  
  
  # plot 3: K-Means error
  fig2 = plt.figure()
  plt.subplot(2,1,1)
  plt.plot(range(2,8), errors)
  plt.xlabel("k")
  plt.ylabel("SSE (Degrees)")
  plt.title(" K-means Clustering: SSE vs k")
  
  # plot 4: K-Means predictions
  plt.subplot(2,1,2)
  plt.plot(range(len(predictions)), predictions, label="Predicted Values")
  plt.subplot(2,1,2)
  plt.plot(range(len(k_targets_16)), k_targets_16, label="Target Values")
  plt.legend(loc=4)
  plt.xlabel("Days")
  plt.ylabel("Temperature (Degrees)")
  plt.title("K-means Clustering : 2016 Output values vs expected values (k=5)")
  
  plt.show()

  
class k_means_cluster():
  def __init__(self, k=5):
    self.k = k
    self.centers = []
    self.clusters = []
    
  def cluster(self, X):
    self.centers = []
    self.clusters = []
    
    # pick k random initial centers
    indices = []
    while len(indices) < self.k:
      j = np.random.randint(low=0, high=len(X)-1)
      if j not in indices:
        indices.append(j)
    
    for i in indices:
      self.centers.append(X[i])
    
    # cluster until there's no change
    prev_centers = ["dummy value"]
    
    iterations = 0
    while (np.any(self.centers != prev_centers)):
      new_clusters = []
      prev_centers = np.array(self.centers)
      
      for i in range(self.k):
        new_clusters.append([])
      
      # cluster data on centers
      for x in X:
        # add x to closest cluster
        nearest_idx = self.closest_center(x)
        new_clusters[nearest_idx].append(x)
        
      # calculate new cluster centers
      for i in range(len(new_clusters)):
        cluster = new_clusters[i]
        center = np.zeros_like(cluster[0])
        
        # average vals for each attribute
        for datum in cluster:
          for j in range(len(datum)):
            center[j] += datum[j]
            
        center = np.divide(center, len(cluster))
        
        # reassign center
        self.centers[i] = center

      # reassign clusters. repeat if no convergence (no more movement)
      self.clusters = new_clusters
      iterations += 1
      
    print self.k, "clusters converged after", iterations, "iterations."
    
  def sse(self, X, targets):
    error = 0
    for x, t in zip(X, targets):
      # find temp of nearest center
      predicted_temp = self.centers[self.closest_center(x)][3]
      
      error += sse(t, predicted_temp)
      
    # error /= len(X)
    return error
  
  def predict(self, X):
    predictions = []
    for x in X:
      # find nearest center
      nearest_idx = self.closest_center(x)
      
      # add the center's max temp 
      predictions.append(self.centers[nearest_idx][3])
      
    return predictions
      
  def euclidian_distance(self, x, center):
    return np.sqrt(np.sum(np.subtract(center, x) ** 2))
    
  def closest_center(self, x):
    min_dist = float("inf")
    min_center = -1
    
    # find the closest cluster center
    for i in range(len(self.centers)):
      center = self.centers[i]
      dist = self.euclidian_distance(x, center)
      
      if (dist < min_dist):
        min_dist = dist
        min_center = i
        
    return min_center
  

class linear_regression():
  def __init__(self, alpha=1E-5, weights=[], w_0=0.0, decay_rate=-1):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate
    
  def predict_regression(self, samples):
    return np.dot(samples, self.weights) + self.w_0

  def train_regression(self, samples, solutions, max_loops=50000):
    err = -1
    i = 0
    output = []
  
    # construct initial weight matrix if needed
    if (self.weights == []):
      self.weights = np.random.randn(len(samples[0]))
      self.w_0 = np.random.randn()
    
    for i in range(max_loops):
      # shuffle samples
      samples, solutions = shuffle_lists(samples, solutions)

      if self.decay_rate > 0 and i % self.decay_rate == 0:
        self.alpha = self.alpha / 2.0
    
      for sample, y in zip(samples, solutions):
        prediction = np.dot(sample, self.weights) + self.w_0
        error = self.error(y, prediction)
        delta = self.alpha * error
        self.weights = np.add(np.multiply(delta, sample), self.weights)
        self.w_0 += delta
        err += sse(y, prediction)
        
      err /= len(samples)
      output.append(err)
      
      print "i=", i, ", err=", err#, ", v_err=", v_err
        
    return output
        
  def error(self, y, y_hat):  
    return (y - y_hat)

def sse(y, y_hat):
  return 0.5 * (y - y_hat) ** 2
  
def load_data(filename, linear=True):
  dataframe = read_csv(filename)
  
  y = dataframe['TMAX']
  if (linear):
    dataframe.pop('TMAX')
  X = dataframe.values
    
  return X, y
  
def shuffle_lists(a, b):
  _a = []
  _b = []
  idx = range(len(a))
  shuffle(idx)
  for i in idx:
    _a.append(a[i])
    _b.append(b[i])
    
  return _a, _b
  
if __name__ == "__main__":
    main()
  