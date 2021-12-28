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
  out = 1./((b-a)**2)*((b*b)/(x*x)-a*a)
  out = tf.where(tf.less(x, 1.0), 0., out)
  out = tf.where(tf.greater(x, b/a), 0., out)
  p = tf.cast(p, tf.float32)
  out = tf.cast(out, tf.float32)
  return out/p

