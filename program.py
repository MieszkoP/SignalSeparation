def ImportLibraries():
  import numpy as np
  import matplotlib.pyplot as plt
  import scipy.stats
  import tensorflow

def GenerateSignal(df, bottom_border, top_border): #Wygeneruj sygnał o ustalonej odległości df. Rozmiar pików losowy (ale w taki spoób aby stosunek wysokości sie zgadzał), miejsce tez losowe (od 400 do 500)
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
