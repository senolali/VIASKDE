# VIASCKDE Index
Python Implementation of VIASKDE Index

This is the python implementation of VIASCKDE Index which is a noval internal clustering validation index and proposed in "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation" by Ali Şenol. 

For using the code you should install KernelDensity. To install it, run "pip install KernelDensity"

VIASCKDE index needs four parameters which are:

X: X={x1, x2,…,xn} ∈ Rd be a dataset containing n points in a d-dimensional space, and xi ∈ Rd.

labels: the predicted labels by the algorithm

kernel: selected kernel method, krnl='gaussian' is te default kernel. But it could be 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.

bandwidth: the bandwidth value of kernel density estimation. b_width=0.05 is the default value. But it could be changed.


According to experimental study, the Gaussian was the best kernel method and the b_width=0.05 was the best bandwidth. 

If you use the code please refer to the citation given below:

"Şenol,A., 2022. "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation", Computational Intelligance and Neurocomputing."


