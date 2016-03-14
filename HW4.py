from random import shuffle
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import numpy as np

def main():
  # get data from csv files
  inputs_15, targets = load_data("2015.csv")
  inputs_16 = load_data("2016.csv")
  
  lr = linear_regression()
  
  # train regression
  output_15, = lr.train_regression(inputs_15, targets)
  
  # predicts outputs
  output_16 = lr.predict_regression(inputs_16)
  
  # plot 1: learning curve
  fig1 = plt.figure()
  plt.plot(output, range(len(output)))
  plt.show()
  
  
class linear_regression():
  def __init__(self, alpha = 0.00001, weights=[], w_0=np.random.randn()):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    
  
  def predict_regression(self, samples):
    output = []
    
    for s in samples:
      output.append(np.dot(self.weights, s) + self.w_0)
      
    return output
  


  def train_regression(self, samples, solutions, max_loops=100, sigma=(1**(-6)), val_ratio=0.2):
    err = -1
    #v_err = -1
    i = 0
    output = []
  
    # construct initial weight matrix if needed
    if (self.weights == []):
      self.weights = np.random.randn(len(samples[0]))
  
    # create validation set
    # samples, solutions = shuffle_lists(samples, solutions)
    # val_idx = (1-val_ratio)*len(samples)
  
    # val_samples = samples[val_idx+1::]
    # val_solutions = solutions[val_idx+1::]
  
    # samples = samples[0:val_idx]
    # solutions = solutions[0:val_idx]
  
    # while (err > 0.8*v_err or i < max_loops):
    print "samples: ", len(samples)
    
    for i in range(max_loops):
      # shuffle samples
      samples, solutions = shuffle_lists(samples, solutions)
    
      for s, y in zip(samples, solutions):
        # run on each sample
        y_hat = np.dot(s, self.weights) + self.w_0
        e = self.sse(y, y_hat)
      
        # update weights
        for w, s in zip(self.weights, samples):
          w -= self.alpha * e * s
        
        # update bias
        self.w_0 -= self.alpha * e
      
        # update error
        err += e
      err /= len(samples)
      output.append(err)
    
      # check validation set
      # for s, y in zip(val_samples, val_solutions):
        # y_hat = np.dot(s, weights)
        # v_err += sse(y, y_hat)
      
      # v_err /= len(val_samples)
    
      print "i=", i, ", err=", err#, ", v_err=", v_err
      # i += 1
        
    return output
        
  def sse(self, y, y_hat):
    # sum squared error
    e = 0
    #for i in range(len(y)):
    e += ((y - y_hat) ** 2)
  
    return (e * 0.5)
  
def load_data(filename):
  dataframe = read_csv(filename)
  print len(dataframe.values)
  
  y = dataframe['TMAX']
  dataframe.pop('TMAX')
  X = dataframe.values
  
 
  
  return X, y
  
def shuffle_lists(a, b):
  # print len(a), ", ", len(b)
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
  