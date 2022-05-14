# The VIASCKDE Index
<h2>Python Implementation of the VIASCKDE Index</h2>

This is the python implementation of VIASCKDE Index which is a noval internal clustering validation index and proposed in "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation" by Ali Şenol. The VIASCKDE index aims to evaluate clusters quality of any clustering algorithm whatever the shape of the clusters are. 
<br><br>
The VIASCKDE index is a kind of index that is not affected by the cluster shape, and thus, it can make a realistic evaluation of clustering performance regardless of the clusters’ shape. Unlike the existing cluster validation indices, our index calculates the compactness and separation values of the cluster based on calculating the compactness and separation values for each data separately. In other words, it calculates the compactness and separation values of the cluster over the distance of data, independent of parameters such as the cluster center because, in non-spherical clusters, the distance of the data to the closest data is more important than its distance to the cluster center. As can be seen in the example given in figure below, the closest data in the cluster that “it belongs to” is used when calculating the compactness value for the data x. Similarly, the separation value of x is calculated by the distance to the closest data of the cluster that “it does not belong”.
<br><br>
<img src="results/fig_5.jpg" width="500"/>
<br><br>
The code given above tries to find the best values of MinPts and &epsilon; which are the parameters of DBSCAN by the help of the VIASCKDE Index in a random search method. The figures given below are the best results that were selected by the VIASCKDE index as the best ones for each dataset.

<img src="results/1_HalfKernel_VIASCKDE.png" width="400"/><img src="results/2_TwoSpirals_VIASCKDE.png" width="400"/><br><img src="results/3_outliers_VIASCKDE.png" width="400"/><img src="results/6_crescentfullmoon_VIASCKDE.png" width="400"/>




<i>VIASCKDE index needs four parameters which are:</i>
<ul>
   <li><b>X</b>: X={x<sub>1</sub>, x2,…,xn} ∈ Rd be a dataset containing n points in a d-dimensional space, and xi ∈ Rd.</li>
   <li><b>labels</b>: the predicted labels by the algorithm</li>
   <li><b>kernel (optional)</b>: selected kernel method, krnl='gaussian' is default kernel. But it could be 'tophat', 'epanechnikov', 'exponential', 'linear', or 'cosine'.</li>
  <li><b>bandwidth(optional)</b>: the bandwidth value of kernel density estimation. b_width=0.05 is the default value. But it could be changed.</li>
 </ul>


According to the experimental studies, the Gaussian was the best kernel method and the b_width=0.05 was the best bandwidth. 
<br><br>
For using the code, you should install KernelDensity. To install it, run "pip install KernelDensity"
<br>
<br>
The VIASCKDE value is expected to be in between [-1, +1], where +1 refers to the best possible value, and -1 refers to the worst possible value.
<br><br>
For more information read the article and if you use the code please cite to the article given below:
<br><br>
"Şenol, A., 2022. "The VIASCKDE Index: A Novel Internal Cluster Validity Index for Arbitrary Shaped Clusters Based on the Kernel Density Estimation", Computational Intelligance and Neurocomputing."


