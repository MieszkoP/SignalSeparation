# Signal Separation
## 0.0.10 version

Version 0.0.10 enables the simplest generation and separation of analytical chemistry signals consisting of two Gaussian peaks 100 wide, 0.8 to 15 high, difference from 0 to 100. The signal length is 1000 points.

The algorithm is based on theoretical considerations about the signal function contained in the paper (see reference 1).

## Features
- Genetic algorithm (see reference 2)
- Quick script for generating 1d-convolutional neural network architecture and training it (the possibility of setting the optimizer, seed, etc. in one line)
- Generation of a simple signal of two overlapping peaks with Gaussian distribution
- Modification of the cost function to compensate for the rarity of the data (see: reference 3)
- **A script that estimates the positions and heights of peaks based on the generated signal (Uses CNN + SVR + Cost Modifications).**


## Installation


```python
pip install signal_separation
```

```python
import signal_separation as s
```

## About ProjektInzynierski.ipynb and program.py

The ProjektInzynierski.ipynb file is an older, less polished, and safer version of the program. It can be run on the Google Colab platform. It uses the program.py file for its operation.

## How to create signal

<img src="https://render.githubusercontent.com/render/math?math=F(t) = G(h_1, t_{m1})%2B G(h_1, t_{m2})%2B N(0, 0.005)">
<img src="https://render.githubusercontent.com/render/math?math=t \in N_0, t<1000 ">

<img src="https://render.githubusercontent.com/render/math?math=N(0, 0.005)"> - a random variable with a normal distribution (mean: 0, std: 0.005)

<img src="https://render.githubusercontent.com/render/math?math=G(h, t)"> - gaussian distribution density function (h - maximum value, t - horizontal coordinate of the maximum value, FWHM = 100)


```python
s.gensign(h1, h2, t1, t2) 
```

for example:

```python
s.gensign(4, 6, 450, 500, return_y_values=False)
```

The resulting signal is the numpy.ndarray array and can be represented by the matplotlib library:

![signal1](https://user-images.githubusercontent.com/78937784/150026502-846d941f-15a2-4423-9642-46ec034cbeaf.png)


## How to estimate signal parameters

In order to estimate, the signal must be provided (it can be simulated) as well as the trained network parameters and trained SVR. 

```python
model = s.create_network(s.only_after_act, 5)
svrh = s.download_svrh()
svrt = s.download_svrt()
s.download_weights(model)
estimation = s.predicting(model, svrh, svrt, signal)
```

Examples of estimated values (h1, h2, t_m1, t_m2): 

```python
>>> estimation
array([[  3.92313289,   6.10226897, 449.80370327, 500.51755778]])
```

![signal2](https://user-images.githubusercontent.com/78937784/150026908-31aaf02b-17f1-4e7a-a942-557d8b3174a8.png)


## License

*Free Software, 
made by Mieszko Pasierbek*

## References

[1] - J. Dubrovkin, „Mathematical methods for separation of overlapping asymmetrical peaks in spectroscopy and chromatography. Case study: one-dimensional signals”, International Journal of Emerging Technologies in Computational and Applied Sciences 11.1 (2015), 1–8.

[2] - T. Nagaoka, „Hyperparameter Optimization for Deep Learning-based Automatic Melanoma Diagnosis System”, Advanced Biomedical Engineering 9 (2020), 225

[3] - M. Steininger, K. Kobs, P. Davidson, A. Krause i A Hotho, „Density-based weighting for imbalanced regression”, Machine Learning 110 (2021), 2187–2211.
