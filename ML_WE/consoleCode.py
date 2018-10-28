Python 3.6.2 (v3.6.2:5fd33b5, Jul  8 2017, 04:57:36) [MSC v.1900 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> dist = [1.5,2.6,7.8,1.4,3.5,2.6,5.1,3.9]
>>> len(dist)
8
>>> labels = np.array([0,0,0,0,1,1,1,1])
>>> np.argsort(dist)
array([3, 0, 1, 5, 4, 7, 6, 2], dtype=int64)
>>> indx = np.argsort(dist)
>>> index
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    index
NameError: name 'index' is not defined
>>> indx
array([3, 0, 1, 5, 4, 7, 6, 2], dtype=int64)
>>> index[:5]
Traceback (most recent call last):
  File "<pyshell#8>", line 1, in <module>
    index[:5]
NameError: name 'index' is not defined
>>> indx[:5]
array([3, 0, 1, 5, 4], dtype=int64)
>>> labels
array([0, 0, 0, 0, 1, 1, 1, 1])
>>> l = labels[3], labels[0], labels[1], labels[5], labels[4]
>>> np.unique(l)
array([0, 1])
>>> np.unique(l, return_counts = True)
(array([0, 1]), array([3, 2], dtype=int64))
>>> count = np.unique(l, return_counts = True)
>>> count[0][np.argmax(count[1])]
0
>>> 
