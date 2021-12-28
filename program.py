import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow

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
