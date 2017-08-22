# Python_Fingerprtint_mpeg7

With this project I created a song recognition algorithm using the MPEG-7 standard for fingerprint calculation.

Requirements
------------

You need:

* python3,
* numpy 
* scipy

install it via:
```
sudo apt-get install python3 python3-venv build-essential libblas-dev \
 liblapack-dev gfortran-4.9 libatlas-dev libatlas-base-dev
```
and:
```
sudo pip3 install numpy
sudo pip3 install scipy
```

Usage:
-----------------
* calculate some fingerprints using:
```
fingerprint = calc_fingerprint(wavefile.wav)
```
* compare to other fingerprint
```
compare_file = 'path/to/fingerprint.csv'  # must be pre calculated before
comparison = np.loadtxt(compare_file, delimiter=",")
result = np.min(euclideanDistance(fingerprint, comparison))
```

compare it to other files and you will find what audio file it is most similar to 
with the over all minimum result 




ToDo:
-----
* finish caomparison with a database


Further information about the algorithm can be found in "Research-Project_Music-Recognition-Using-Python_Konstantin-Brand.pdf"
