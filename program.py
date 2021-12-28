import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf

def GenerateSignal(df, bottom_border, top_border): 
  '''Generate a signal with a fixed distance df. Uniform random peak size (from bottom_border to top_border), also uniform random location (from 400 to 500)'''
  height1 = np.random.uniform(bottom_border,top_border)
  height2 = np.random.uniform(bottom_border,top_border)
  x = np.arange(0,1000,1)
  place1 = np.random.uniform(400,500)
  place2 = place1+df
  y_f1 = scipy.stats.norm(place1, 100/2.355)
  y1 = y_f1.pdf(x)*100/2.355*(2*3.14)**(1/2)*height1
  y_f2 = scipy.stats.norm(place2, 100/2.355)
  y2 = y_f2.pdf(x)*100/2.355*(2*3.14)**(1/2)*height2
  y = y1+y2
  y = np.random.normal(loc=0, scale=0.005, size=1000)+y
  return y, height1, height2, place1, place2

def RatioOfUniforms(a,b,x,p=1.0):
  '''Generate a function describing the probability density of the ratio of the variables generated in a normal distribution (normal distribution range from a to b).'''
  out = 1./((b-a)**2)*((b*b)/(x*x)-a*a)
  out = tf.where(tf.less(x, 1.0), 0., out)
  out = tf.where(tf.greater(x, b/a), 0., out)
  p = tf.cast(p, tf.float32)
  out = tf.cast(out, tf.float32)
  return out/p

def SumOfUniforms(a,b,x,p=1.0):
  '''Generate a function describing the probability density of the sum of the variables generated in the normal distribution (normal distribution range from a to b).'''
  out = 0
  out = tf.where(tf.greater(x,a), 4/(b-a)**2*(x-a), out)
  out = tf.where(tf.greater(x,(b-a)/2+a), -4/(b-a)**2*(x-b), out)
  out = tf.where(tf.greater(x,b), 0.0, out)
  p = tf.cast(p, tf.float32)
  out = tf.cast(out, tf.float32)
  return out/p

def Standardize(Y, x1_min, x1_max, x2_min, x2_max, y_min, y_max):
  '''# Standardize the dependent variables so that the values are between 0.1 and 0.9'''
  size = len(Y)

  delta_x1 = x1_max - x1_min
  delta_x2 = x2_max - x2_min

  Y_new = np.zeros((size, 4))
  Y_new[:,:2] = (Y[:,:2]-y_min)/(y_max-y_min)*0.8+0.1
  Y_new[:,2] = (((Y[:,2])-x1_min)/delta_x1)*0.8+0.1
  Y_new[:,3] = (((Y[:,3])-x2_min)/delta_x2)*0.8+0.1
  return Y_new

def DeStandarize(Y, x1_min, x1_max, x2_min, x2_max, y_min, y_max):
  '''Destandarize the dependent variables from 0.1 - 0.9 to correct'''
  delta_x1 = x1_max - x1_min
  delta_x2 = x2_max - x2_min

  if len(Y.shape)==1:
    Y = Y.reshape(1,4)

  size = len(Y)
  Y_old = np.zeros((size, 4))

  Y_old[:,:2] = ((Y[:,:2]-0.1)/0.8)*(y_max-y_min)+y_min
  Y_old[:,2] = ((Y[:,2]-0.1)/0.8)*(delta_x1)+x1_min
  Y_old[:,3] = ((Y[:,3]-0.1)/0.8)*(delta_x2)+x2_min
  
  return Y_old

def DeStandarizeHeight(h, y_min, y_max):
  return ((h-0.1)/0.8)*(y_max-y_min)+y_min

def PlacesChange(p, p_min, p_max):
  return ((p-0.1)/0.8)*(p_max-p_min)+p_min

def MakeHistogram(data, name, range=None):
  n, bins, patches = plt.hist(data, bins=50, range=range, density=True)
  plt.ylabel("Gęstość częstości/prawdopodobieństwa")
  width = (bins[-1]-bins[0]) / 50
  width = round(width, 2)
  plt.xlabel(name+", (Szerokosć słupka: "+str(width)+")")
  return n

def HeightProportionTensorflow(x,y):
  output = tf.where(tf.greater(x/y, 1.0), x/y, y/x)
  output = tf.where(tf.greater(output, 20.0), 20.0, output)
  return output

def CostFunctionNominator(function, a, b, alpha):
  p1 = function(a,b,1.)
  p2 = function(a,b,(a+b)/2)
  p = tf.where(tf.greater(p1,p2), p1, p2)
  p = tf.cast(p, tf.float32)

  def out(m, d=1.):
    return (1-alpha*function(a,b,m,p=p))/d

  return out

def CostFunctionDenominator(function, a,b,alpha, y):
  out = CostFunctionNominator(function, a,b,alpha=alpha)
  out = tf.math.reduce_mean(out(y), 0)
  return out 

def CostFunction(function, a,b,alpha, y):

  out = CostFunctionNominator(function, a, b, alpha)
  denominator = CostFunctionDenominator(function, a, b, alpha, y)
  denominator = tf.cast(denominator, tf.float32)

  def out_final(y_data):
    return out(y_data, d=denominator)

  return out_final

def Uniform(a,b,x):
  out = 0
  out = tf.where(tf.greater(x,a), 1/(b-a), out)
  out = tf.where(tf.greater(x,b), 0, out)

  return out

class Populaton:
  
  def __init__(self, number, n_hip):
    self.number = number
    self.n_hip = n_hip
    self.h = list(range(n_hip+1))
    self.h[0] = ['nr:', list(range(1,number+1))]
    self.stored = np.zeros((0,n_hip+2))

  def define_feature(self, place, name, l):
    #print(self.f)
    self.h[place] = [name, l]
    #print(self.f)

  def initialize(self): #inicjalizacja (poprzez losowanie) wartości poczatkowych
    self.initialized = np.zeros((self.number,self.n_hip+2))
    self.initialized[:,0] = range(1,self.number+1)
    for i in range(1,self.n_hip+1):
      self.initialized[:,i] = np.random.randint(0,len(self.h[i][1]), self.number)
    print(self.initialized)
    self.it=0

  def read(self, hiper):
    place = self.initialized[self.it, hiper]
    #print(place)
    out = self.h[hiper][1][round(place)]
    return out

  def isFinished(self):
    return self.it<self.number

  def nextInd(self):
    self.it+=1

  def IndBeg(self):
    self.it=0

  def ReadMetric(self, Value): #Przepisz wartość metryki do parametru osobnika
    self.initialized[self.it,self.n_hip+1] = Value

  def Store(self): #Przechowywanie wyników wcześniejszych ewaluacji
    self.stored = np.append(self.stored, self.initialized, axis=0)

  def Save(self): #Zapisanie danych
    np.savetxt('DataStored.csv', self.stored)
    np.savetxt('DataCurrent.csv', self.initialized)

  def Load(self, stored_datafile, current_datafiled):
    self.stored = np.loadtxt(stored_datafile)
    self.initialized = np.loadtxt(current_datafiled)

  def Sorting(self):
    self.initialized = self.initialized[self.initialized[:, -1].argsort()]

  def Selection(self):
    self.initialized[:-2,:] = self.initialized[1:-1,:]
    self.initialized[-2,:] = self.initialized[-1,:]

  def Crossing(self):
    self.cross_position = np.random.randint(2,1+self.n_hip)
    self.initialized[:2,self.cross_position:] = self.initialized[1::-1,self.cross_position:]

    self.cross_position = np.random.randint(2,1+self.n_hip)
    self.initialized[2:4,self.cross_position:] = self.initialized[3:1:-1,self.cross_position:]

  def Mutation(self, m_rate):
    self.initialized[:,0] = range(1,self.number+1)
    self.initialized[:,-1] = [0]
    n_mutations = int(np.ceil(m_rate*self.n_hip*self.number)) #Policzenie ilości dokonywanych mutacji
    for i in range(n_mutations):
      x = np.random.randint(1,self.n_hip+1)
      y = np.random.randint(self.number)
      #print(x +str(" ")+y)
      self.initialized[y,x] = np.random.randint(0,len(self.h[x][1]))

  def Check(self):
    for y in range(len(self.stored)):
      for yi in range(len(self.initialized)):
        if (self.initialized[yi,1:-1]==self.stored[y,1:-1]).all():
          self.initialized[yi,-1] = self.stored[y,-1]

  def CheckIsMetricWrote(self):
    return self.initialized[self.it,self.n_hip+1]!=0
  
from tensorflow.keras import datasets, layers, models, losses, Model, initializers

def MultiplyCNN(n,karnel_size,added_filters, filter_beg,dense1,dense2,x, BatchN):
  for i in np.arange(n):
    x = layers.Conv1D(i*added_filters+filter_beg,karnel_size, 1) (x)

    if BatchN()[1] == True:
      x = layers.BatchNormalization()(x)
    x = layers.Activation('softplus')(x)
    if BatchN()[2] == True:
      x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(2)(x)
  out = layers.Flatten() (x)
  out = layers.Dense(dense1, activation='softplus') (out)
  out = layers.Dense(dense2, activation='sigmoid') (out)
  return out



def MultiplyCNNDenseless(n,karnel_size,added_filters,filter_beg,x,BatchN):
  for i in np.arange(n):
    x = layers.Conv1D(i*added_filters+filter_beg,karnel_size, 1) (x)

    if BatchN()[1] == True:
      x = layers.BatchNormalization()(x)

    x = layers.Activation('softplus')(x)
    if BatchN()[2] == True:
      x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(2)(x)
  return x, (n-1)*added_filters+filter_beg

from keras import backend

def mean_squared_error_new_height(y_true, y_pred, y_true1, y_true2, weightfunction):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)

  y_true1 = DeStandarizeHeight(y_true1, 0.8, 15)
  y_true2 = DeStandarizeHeight(y_true2, 0.8, 15)
  
  return backend.mean(tf.math.squared_difference(y_pred, y_true)*weightfunction(HeightProportionTensorflow(y_true1, y_true2)), axis=-1)

def mean_squared_error_new_position2(y_true, y_pred, weightfunction):
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  
  return backend.mean(tf.math.squared_difference(y_pred, y_true)*weightfunction(y_true), axis=-1)

def height_dif(y_true, y_pred):
  return mean_squared_error_new_height(HeightProportionTensorflow(y_true[:,1], y_true[:,0]),HeightProportionTensorflow(y_pred[:,1],y_pred[:,0]), y_true[:,0], y_true[:,1])*0.00255

def freq_dif(y_true, y_pred):

  true = PlacesChange(y_true[:,3], 0.1, 1.7)-y_true[:,2]
  pred = PlacesChange(y_pred[:,3], 0.1, 1.7)-y_pred[:,2]
  return tf.keras.losses.mean_squared_error(true,pred)*0.224

def h1loss(y_true, y_pred):
  return tf.keras.losses.mean_squared_error(y_true[:,0],y_pred[:,0])

def h2loss(y_true, y_pred):
  return tf.keras.losses.mean_squared_error(y_true[:,1],y_pred[:,1])

def p1loss(y_true, y_pred):
  return tf.keras.losses.mean_squared_error(y_true[:,2],y_pred[:,2])

def p2loss(y_true, y_pred):
  return mean_squared_error_new_position2(y_true[:,3],y_pred[:,3])

def custom_loss_function(y_true, y_pred):
  loss = height_dif(y_true, y_pred)+h1loss(y_true, y_pred)+h2loss(y_true, y_pred)+p1loss(y_true, y_pred)+p2loss(y_true, y_pred)+freq_dif(y_true, y_pred)
  return loss

def NoBatch():
  return [False, False, False]

def OnlyBeforeAct():
  return [False, True, False]

def OnlyAfterAct():
  return [False, False, True]

def AfterInput():
  return [True, False, False]

def AfterInputAndAfterAct():
  return [True, False, True]

def AfterInputAndBeforeAct():
  return [True, True, False]

def TrainNetwork(seed, fork_point, optim, basic_lr, BatchNormalization):
  tf.random.set_seed(seed)
  
  model = CreateNetwork(BatchNormalization, fork_point) #Ustal czy w architekturze sieci będzie normalizacja Batch na początku czy nie

  opt = optim(learning_rate=basic_lr)
  model.compile(loss=custom_loss_function,
                optimizer=opt,
                metrics = [h1loss, h2loss, p1loss, p2loss, height_dif, freq_dif, r_square],
                run_eagerly=False)

  earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='min')
  reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1, min_lr=0.00001, min_delta=1e-4, mode='min')

  mymodel=model.fit(
      X,
      Y_new,
      batch_size=100,
      epochs=50,
      verbose=1,
      callbacks=[reduce_lr_loss, earlyStopping],
      validation_data = (X_t, Y_new_test)
  )

  print('Koniec uczenia')
  return mymodel

