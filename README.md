# VIASCKDE Index
<h2>Python Implementation of VIASKDE Index</h2>

This is the python implementation of VIASCKDE Index which is a noval internal clustering validation index and proposed in "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation" by Ali Şenol. The VIASCKDE index aims to evaluate clusters quality of any clustering algorithm whatever the shape of the clusters are. The code given above tries to find the best values of MinPts and epsilon which are the parameters of DBSCAN by the help of VIASCKDE Index in a random search method. The figures given below are the best results that were selected by VIASCKDE index as the best ones for each dataset.

<img src="results/1_HalfKernel_VIASCKDE.png" width="400"/><img src="results/2_TwoSpirals_VIASCKDE.png" width="400"/><br><img src="results/3_outliers_VIASCKDE.png" width="400"/><img src="results/6_crescentfullmoon_VIASCKDE.png" width="400"/>



<i>VIASCKDE index needs four parameters which are:</i>
<ul>
   <li><b>X</b>: X={x1, x2,…,xn} ∈ Rd be a dataset containing n points in a d-dimensional space, and xi ∈ Rd.</li>
   <li><b>labels</b>: the predicted labels by the algorithm</li>
   <li><b>kernel</b>: selected kernel method, krnl='gaussian' is te default kernel. But it could be 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.</li>
  <li><b>bandwidth</b>: the bandwidth value of kernel density estimation. b_width=0.05 is the default value. But it could be changed.</li>
 </ul>



<br>
For using the code, you should install KernelDensity. To install it, run "pip install KernelDensity"
<br>
<br>


According to the experimental studies, the Gaussian was the best kernel method and the b_width=0.05 was the best bandwidth. 
<br><br>
The VIASCKDE value is expected to be in between [-1, +1], where +1 refers to the best possible value, and -1 refers to the worst possible value.
<br><br>
If you use the code please refer to the citation given below:
<br><br>
"Şenol,A., 2022. "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation", Computational Intelligance and Neurocomputing."


