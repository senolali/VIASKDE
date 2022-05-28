import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score
import random
from scipy.spatial import KDTree
import scipy.io
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
from IPython import get_ipython
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")
get_ipython().magic('reset -sf')
get_ipython().magic('clear all -sf')

def closest_node(n, v):
    kdtree = KDTree(v)
    d, i = kdtree.query(n)
    return d

def VIASCKDE(X, labels,krnl='gaussian', b_width=0.05):  
    CoSeD=np.array([],[])
    num_k = np.unique(labels)
    kde = KernelDensity(kernel=krnl, bandwidth=b_width).fit(X)
    iso = kde.score_samples(X)

    ASC=np.array([])
    numC=np.array([])
    CoSeD=np.array([])
    viasc=0
    if len(num_k)>1:      
        for i in num_k:
            data_of_cluster=X[labels==i]    
            data_of_not_its=X[labels!=i]
            isos=iso[labels==i]
            isos = (isos-min(isos))/(max(isos)-min(isos))
            for j in range(len(data_of_cluster)): # for each data of cluster j
                row=np.delete(data_of_cluster,j,0) # exclude the data j 
                XX=data_of_cluster[j]
                a=closest_node(XX, row)
                b=closest_node(XX, data_of_not_its)
                ASC=np.hstack((ASC,((b-a)/max(a,b)) * isos[j]))
            numC=np.hstack((numC,ASC.size))
            CoSeD=np.hstack((CoSeD,ASC.mean())) 
        for k in range(len(numC)):
            viasc+=numC[k]*CoSeD[k];
        viasc=viasc/sum(numC)
        print("viasc=%0.4f"%viasc)
    else:
        viasc=float("nan")
    return viasc
def plotGraph(data, labels):
    plt.rcParams['figure.dpi'] = 100
    plt.subplots_adjust(left=.2, right=.98, bottom=.001, top=.96, wspace=.05,
                hspace=.01)
    plt.rcParams["figure.figsize"] = (4.2,4)
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors,edgecolors='k')

data_sets={1,2,3,4,5,6}

for dataset in data_sets:
    print("dataset=",dataset)
    plotFigure=1
    loop=0
    if (dataset==1):
        data = scipy.io.loadmat("Datasets/halfkernel.mat")
        data=data["halkernel"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="1_HalfKernel_"
    elif dataset==2:
        data = pd.read_csv("Datasets/TS_labels.csv")
        labels_true = np.mat(data)
        labels_true=np.ravel(labels_true)
        data = pd.read_csv("Datasets/TS_X.csv")
        X = np.mat(data)
        dataset_name="2_TwoSpirals_"
    elif dataset==3:
        data = np.genfromtxt("Datasets/outlers700.txt", delimiter=',', dtype=float,usecols=range(3))
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="3_outliers_"
    elif dataset==4:
        X = np.loadtxt("Datasets/zelnik5.txt", delimiter=',', dtype=float)
        labels_true= np.loadtxt("Datasets/zelnik5_label.txt", delimiter=',', dtype=float)
        dataset_name="4_zelnik5_"
    elif dataset==5:
        data = scipy.io.loadmat("Datasets/clusterin.mat")
        data=data["clusterincluster"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="5_clusterincluster_"
    elif dataset==6:
        data = scipy.io.loadmat("Datasets/crescentfullmoon.mat")
        data=data["crescentfullmoon"];
        labels_true = data[:,2]
        X = data[:,0:2]
        dataset_name="6_crescentfullmoon_"


    ####MinMaxNormalization#######################################################
    scaler = MinMaxScaler()
    scaler.fit(X)
    MinMaxScaler()
    X=scaler.transform(X)
    
    maxVIASCKDE=float('-inf')
    while loop<100: 
        loop+=1
        eps=random.uniform(0.05, 0.1) #select DBSCAN parameter randomly
        min_samples=random.randint(5, 25) #select DBSCAN parameter randomly
        print("Eps=%0.4f, MinPts=%d"%(eps,min_samples))
        # Fitting DBSCAN to the dataset and predict the Cluster label    
        db = DBSCAN(eps, min_samples)
        labels = db.fit_predict(X) 
        labels=np.ravel(labels)
      
        ARI=adjusted_rand_score(labels_true, labels);
        print("%d. ARI=%0.4f" %(loop,ARI))
        plotGraph(X,labels)
        eps=str("%0.2f, MinPts=%d} => {ARI=%0.4f}"%(eps,min_samples,ARI)) 
        s="DBSCAN {\u03B5="+eps
        plt.title(s,size=10)

        num_k = np.unique(labels)
        if len(num_k)>1:
            viasckde=VIASCKDE(X,labels)  # VIASCKDE index validation
            left, width = .25, .5
            bottom, height = .25, .5
            plt.subplots_adjust(bottom=0.25) 
            text = plt.text(0.1, -0.2, 
                            str("VIASCKDE=%.4f"%viasckde), 
            horizontalalignment='center', wrap=True ) 
            plt.tight_layout(rect=(0,.05,1,1)) 
            text.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='black'))
            if viasckde>maxVIASCKDE:
                maxVIASCKDE=viasckde
                VIASCKDE_ARI=ARI 
                plt_name=str("results/"+dataset_name+"VIASCKDE.png")
                plt.savefig(plt_name)
            plt.show()
            plt.clf()
    print("VIASCKDE=%0.4f, ARI=%0.4f"%(maxVIASCKDE,VIASCKDE_ARI))
    
    file=str("results/"+dataset_name+".txt")
    open(file, "w").close()
    f = open(file, "a")
    f.write("VIASCKDE=%0.4f, ARI=%0.4f\n"%(maxVIASCKDE,VIASCKDE_ARI))
    f.close()

   