from random import shuffle
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
import numpy as np
import math

def main():
  # get data from csv files
  inputs_15, targets_15 = load_data("./2015.csv")
  inputs_16, targets_16 = load_data("./2016.csv")
  
  NUM_RUNS = 10
  
  # train regression

  bestError = -1
  bestFit = None
  bestTrain = None

  for _ in range(NUM_RUNS):

    lr = linear_regression(alpha=1E-3, decay_rate=800, weights=np.random.randn(len(inputs_15[0])), spike=[4000, 6000])

    error_15 = lr.train_regression(inputs_15, targets_15, max_loops=int(5E4))
    
    # predicts outputs
    predictions_16 = lr.predict_regression(inputs_16)
    error_16 = np.sum((predictions_16 - targets_16) ** 2)

    """
    if bestError == -1 or error_15[-1] < bestError:
      bestError = error_15[-1]
      bestFit = predictions_16
      bestTrain = error_15
      """

    # plot 1: learning curv
    fig1 = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(range(len(error_15)), error_15, label="Error")
    plt.legend()
    plt.xlabel("Training Iteration")
    plt.ylabel("Error magnitude")
    plt.title("Error progression through training")
    
    plt.subplot(2,1,2)
    plt.plot(range(len(predictions_16)), predictions_16, label="Predicted Values 2016")
    plt.subplot(2,1,2)
    plt.plot(range(len(targets_16)), targets_16, label="Target Values 2016")
    plt.legend()
    plt.title("Prediction comparison, 2016")
    plt.xlabel("Sample number")
    plt.ylabel("TMAX Value")
    plt.show(block=False)
  plt.show()

  
  
class linear_regression():
  def __init__(self, alpha = 1E-4, weights=[], w_0=0.0, decay_rate=-1, spike=[], epsilon = 0.000001):
    self.alpha = alpha
    self.weights = weights
    self.w_0 = w_0
    self.decay_rate = decay_rate
    self.spike = spike
    self.epsilon = epsilon
  
  def predict_regression(self, samples):
    return np.dot(samples, self.weights) + self.w_0

  def train_regression(self, samples, solutions, max_loops=10000, sigma=(1**(-6)), val_ratio=0.2):
    
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
      if (not self.decay_rate == -1) and i % self.decay_rate == 0:
        self.alpha = self.alpha / 2.0

      if (not len(self.spike) == 0) and (i % self.spike[0] == 0):
        self.prevAlpha = self.alpha
        self.alpha = 0.001
      if (not len(self.spike) == 0) and (i % (self.spike[0] + 2) == 0):
        self.alpha = self.prevAlpha
        del self.spike[0]


      samples, solutions = shuffle_lists(samples, solutions)
      err = 0.0
      for sample, y in zip(samples, solutions):
        prediction = np.dot(sample, self.weights) + self.w_0
        error = self.sse(y, prediction)
        delta = self.alpha * error
        self.weights += delta
        self.w_0 += delta
        err += error

        """
        # run on each sample
        y_hat = np.dot(sample, self.weights) + self.w_0
        e = self.sse(y, y_hat)
        
        # update weights
        #print 'Weights before : ', self.weights
        newWeights = []
        for w, solution in zip(self.weights, solutions):
          newWeight = w + self.alpha * e * solution
          #print 'With beginning weight of ', w, ' and error of ', e, ' and solution of ', solution,' calculated new weight of ', newWeight
          #raw_input()
          newWeights.append(w + self.alpha * e * solution)
        
        #print 'Weights after : ', newWeights
        
        self.weights = newWeights
      
        
      
        # update error
        err += e
        """
        
      err /= len(samples)
      err = math.fabs(err)
      output.append(err)
      if err < self.epsilon:
        break
      
    
      # check validation set
      # for s, y in zip(val_samples, val_solutions):
        # y_hat = np.dot(s, weights)
        # v_err += sse(y, y_hat)
      
      # v_err /= len(val_samples)
    
      #print "i=", i, ", err=", err#, ", v_err=", v_err
      # i += 1
        
    return output
        
  def sse(self, y, y_hat):
    #print y, y_hat
    # sum squared error
    e = 0
    #for i in range(len(y)):
    e += 0.5 * (y - y_hat)
        #e += math.sqrt(((y - y_hat) ** 2))
  
    return (e * 0.5)
  
def load_data(filename):
  dataframe = read_csv(filename)
  
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
  