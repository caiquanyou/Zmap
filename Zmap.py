from .util import *
from .model import cell2clusters,cell2spots,cell2spots3D
from .ot_model import solve_OT
import lap
import pandas as pd
from scipy.sparse import issparse
import numpy as np
import scanpy as sc
import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cos_s
class Zmap:
    """
    This class is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset(or bulk RNA datasets).
    It will use the Zmap algorithm on the scRNA-seq dataset and reconstructe a new spatial transcriptomics data.
    The input parameters are the scRNA-seq dataset (scadata), the spatial transcriptomics dataset (stdata) or the bulk datasets (bulkX and bulkY), and the genes to be used (genes).
    """
    def __init__(
        self, 
        scdata, 
        stdata=None,
        bulkX=None, 
        bulkY=None, 
        genes=None, 
        histology=None, 
        cluster_time=1, 
        custom_label=None,
        pca=None, 
        n_pcs=50, 
        pc_frac=0.5, 
        target_num = 10,
        cluster_thres=None,
        ot_cluster_thres=0.01,
        emptygrid=None,
        shape="square", 
        device='cpu'):
        """
        Args:
            scdata (anndata.AnnData): scRNA-seq data
            stdata (anndata.AnnData): spatial transcriptomics data
            bulkX (anndata.AnnData): bulk transcriptomics data, default is None
            bulkY (anndata.AnnData): bulk transcriptomics data, default is None
            genes (list): genes to be used for mapping
            histology (np.array): histology data, default is None
            cluster_time (int): numbers to run clustering
            custom_label (str): label for the cluster
            pca (str): pca method to be used,default is None
            n_pcs (int): number of pcs to be used
            pc_frac (float): fraction of pcs to be used
            target_num (int): number of clusters
            cluster_thres (float): thres to filter small values
            emptygrid (np.array): empty grid matrix, 
            shape (str): shape of the spatial transcriptomics data, default is "square", other option is "hexagon"
            device (str): device to be used for training, default is 'cpu'
        """
        self.scdata = scdata
        self.stdata = stdata
        self.bulkX = bulkX
        self.scLocX = None
        self.bulkY = bulkY
        self.scLocY = None
        self.emptygrid = emptygrid
        self.genes = genes
        self.histology = histology
        self.cluster_time = cluster_time
        self.custom_label = custom_label
        self.cluster_label = None
        self.pca = pca
        self.n_pcs = n_pcs
        self.pc_frac = pc_frac
        self.target_num = target_num
        self.cluster_thres = cluster_thres
        self.ot_cluster_thres = ot_cluster_thres
        self.cluster_matrix = None
        self.spot_matrix = None
        self.shape = shape
        self.device = device

    def allocate(self):
        """
        This function is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset. 
        The input parameters are the scRNA-seq dataset (scadata), the spatial transcriptomics dataset (stdata), the label for cell clusters (label), and the genes to be used for training (genes).
        The output is a matrix with the same shape as the spatial transcriptomics data, where each element is a probability of the corresponding cell belonging to a specific cluster.
        """
        if self.stdata is not None:
            self.bulkX = generate_Xstrips(self.stdata)
            self.bulkY = generate_Ystrips(self.stdata)
            self.cluster_label = []
            if self.custom_label is not None:
                self.cluster_label.append(self.custom_label)
                cluster_time = self.cluster_time - 1
                self.cluster_matrix = cluster_mapping(self.scdata,self.stdata,self.genes,label=self.custom_label,device = self.device,thres=self.cluster_thres)
                if self.cluster_thres is not None:
                    self.cluster_matrix[self.cluster_matrix<0.01]=-1e15
            else:
                # self.cluster_label.append('All_pcs')
                # cluster_time = self.cluster_time - 1
                # random_cluster(self.stdata,label='All_pcs',n_pcs=self.n_pcs,pc_frac=1.0,samples_time=1,shape=self.shape,target_num = self.target_num)
                # self.cluster_matrix = cluster_mapping(self.scdata,self.stdata,self.genes,label='All_pcs',device = self.device,thres=self.cluster_thres)
                cluster_time = self.cluster_time
                self.cluster_matrix = 0
            if cluster_time == 0:
                self.spot_matrix = spot_mapping(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)
            else:
                for i in range(cluster_time):
                    label = 'clutimes_' + str(i)
                    self.cluster_label.append(label)
                    print("Start to run the %dth clustering."%i)
                    random_cluster(self.stdata,label,n_pcs=self.n_pcs,pc_frac=self.pc_frac,samples_time=self.cluster_time,shape=self.shape,target_num = self.target_num)
                    print("Start to run the %dth mapping."%i)
                    self.cluster_matrix += cluster_mapping(self.scdata,self.stdata,self.genes,label=label,device = self.device,thres=self.cluster_thres)
                self.cluster_matrix = self.cluster_matrix/self.cluster_time
                if self.cluster_thres is not None:
                    self.cluster_matrix[self.cluster_matrix<0.01]=-1e15
                print("Start to run the spot mapping.")
                self.spot_matrix = spot_mapping(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)
        else:
            self.spot_matrix = spot_mapping(self.scdata,None,self.cluster_matrix.values,self.bulkX,self.bulkY,genes=self.genes,device = self.device)
    
    def ot_allocate(self):
        """
        This function is used to run the optimal transport algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset.
        """
        if self.stdata is not None:
            self.bulkX = sc.AnnData(X=generate_Xstrips(self.stdata),var=self.stdata.var)
            self.bulkY = sc.AnnData(X=generate_Ystrips(self.stdata),var=self.stdata.var)
            self.scLocX = strips_mapping(self.scdata,self.bulkX,self.genes,device = self.device,thres=self.ot_cluster_thres)
            self.scLocY = strips_mapping(self.scdata,self.bulkY,self.genes,device = self.device,thres=self.ot_cluster_thres)
            self.spot_matrix = solve_OT(self.emptygrid,self.scLocX,self.scLocY,thres=self.ot_cluster_thres,njob=8)
        else:
            self.scLocX = strips_mapping(self.scdata,self.bulkX,self.genes,device = self.device,thres=self.ot_cluster_thres)
            self.scLocY = strips_mapping(self.scdata,self.bulkY,self.genes,device = self.device,thres=self.ot_cluster_thres)
            self.spot_matrix = solve_OT(self.emptygrid,self.scLocX,self.scLocY,thres=self.ot_cluster_thres,njob=8)

"""
    For Visium mapping.
"""
class Zmap3D:
    """
    This class is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset(or bulk RNA datasets).
    It will use the Zmap algorithm on the scRNA-seq dataset and reconstructe a new spatial transcriptomics data.
    The input parameters are the scRNA-seq dataset (scadata), the spatial transcriptomics dataset (stdata) or the bulk datasets (bulkX and bulkY), and the genes to be used (genes).
    """
    def __init__(
        self, 
        scdata, 
        stdata,
        genes=None, 
        cluster_time=1, 
        custom_label=None,
        pca=None, 
        n_pcs=50, 
        pc_frac=0.5, 
        target_num = 10,
        cluster_thres=None,
        shape="hexagon", 
        device='cpu'):
        """
        Args:
            scdata (anndata.AnnData): scRNA-seq data
            stdata (anndata.AnnData): spatial transcriptomics data
            genes (list): genes to be used for mapping
            cluster_time (int): numbers to run clustering
            custom_label (str): label for the cluster
            pca (str): pca method to be used,default is None
            n_pcs (int): number of pcs to be used
            pc_frac (float): fraction of pcs to be used
            target_num (int): number of clusters
            cluster_thres (float): thres to filter small values
            shape (str): shape of the spatial transcriptomics data, default is "square", other option is "hexagon"
            device (str): device to be used for training, default is 'cpu'
        """
        self.scdata = scdata
        self.stdata = stdata
        self.genes = genes
        self.cluster_time = cluster_time
        self.custom_label = custom_label
        self.cluster_label = None
        self.pca = pca
        self.n_pcs = n_pcs
        self.pc_frac = pc_frac
        self.target_num = target_num
        self.cluster_thres = cluster_thres
        self.cluster_matrix = None
        self.spot_matrix = None
        self.shape = shape
        self.device = device

    def allocate(self):
        """
        This function is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset. 
        The input parameters are the scRNA-seq dataset (scadata), the spatial transcriptomics dataset (stdata), the label for cell clusters (label), and the genes to be used for training (genes).
        The output is a matrix with the same shape as the spatial transcriptomics data, where each element is a probability of the corresponding cell belonging to a specific cluster.
        """
        self.bulkX = generate_Xstrips(self.stdata)
        self.bulkY = generate_Ystrips(self.stdata)
        self.bulkY = generate_Zstrips(self.stdata)
        self.cluster_label = []
        if self.custom_label is not None:
            self.cluster_label.append(self.custom_label)
            cluster_time = self.cluster_time - 1
            self.cluster_matrix = cluster_mapping(self.scdata,self.stdata,self.genes,label=self.custom_label,device = self.device,thres=self.cluster_thres)
            if self.cluster_thres is not None:
                self.cluster_matrix[self.cluster_matrix<0.01]=-1e15
        else:
            cluster_time = self.cluster_time
            self.cluster_matrix = 0
        if cluster_time == 0:
            self.spot_matrix = spot_mapping3D(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)
        else:
            for i in range(cluster_time):
                label = 'clutimes_' + str(i)
                self.cluster_label.append(label)
                random_cluster(self.stdata,label,n_pcs=self.n_pcs,pc_frac=self.pc_frac,samples_time=self.cluster_time,shape=self.shape,target_num = self.target_num)
                self.cluster_matrix += cluster_mapping(self.scdata,self.stdata,self.genes,label=label,device = self.device,thres=self.cluster_thres)
            self.cluster_matrix = self.cluster_matrix/self.cluster_time
            if self.cluster_thres is not None:
                self.cluster_matrix[self.cluster_matrix<0.01]=-1e15
            self.spot_matrix = spot_mapping3D(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)

def strips_mapping(
    scadata,
    bulk,
    genes=None,
    device = 'cpu',
    num_epochs = 500,
    learning_rate = 0.1,
    thres=None):
    """
    this function is to map the scRNA-seq data to the strips bulk transcriptomics data
    Args:
        scadata (anndata.AnnData): scRNA-seq data
        bulk (anndata.AnnData): bulk transcriptomics data
        genes (list): genes to be used for mapping
        device (str): device to be used for training
        num_epochs (int): number of epochs to be used for training
        learning_rate (float): learning rate to be used for training
        thres (floot): thres to filter small values
    Return:
        bulk_matrix (np.array): sc to bulk mapping matrix
    """
    if genes is None:
        preprocess(scadata,bulk)
    else:
        preprocess(scadata,bulk,genes)
    overlap_genes = scadata.uns["overlap"]
    S = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32",)
    G = np.array(bulk[:, overlap_genes].X, dtype="float32")
    bulk_mapper = cell2clusters(scdata=S,clusters=G,device=device)
    bulk_matrix = bulk_mapper.fit(num_epochs=num_epochs,learning_rate=learning_rate,print_each=100)
    if thres == None:
        return bulk_matrix
    else:
        bulk_matrix[bulk_matrix<thres]=0
        return bulk_matrix

def cluster_mapping(
    scadata,
    stdata,
    genes=None,
    label='leiden',
    device = 'cpu',
    num_epochs = 500,
    learning_rate = 0.1,
    thres=None):
    """
    Mapping the scRNA-seq data to the cluster transcriptomics data
    Args:
        scadata (anndata.AnnData): scRNA-seq data
        stdata (anndata.AnnData): spatial transcriptomics data
        genes (list): genes to be used for mapping
        label (str): clusters label stored in the spatial data obs
        device (str): device to be used for training
        num_epochs (int): number of epochs to be used for training
        learning_rate (float): learning rate to be used for training
        thres (floot): thres to filter small values
    Return:
        cluster_mapper_matrix (np.array): sc to cluster mapping matrix
    """
    cluster_bulk = generate_cluster_exp(stdata,label=label)
    cluster_bulk = sc.AnnData(X=cluster_bulk,var=stdata.var)
    if genes is None:
        preprocess(scadata,cluster_bulk)
    else:
        preprocess(scadata,cluster_bulk,genes)
    overlap_genes = scadata.uns["overlap"]
    scdata = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32",)
    clusters = np.array(cluster_bulk[:, overlap_genes].X, dtype="float32")
    cluster_mapper = cell2clusters(scdata=scdata,clusters=clusters,device=device)
    cluster_mapper_matrix = cluster_mapper.fit(num_epochs=num_epochs,learning_rate=learning_rate,print_each=100)
    cell_clu = transform(stdata.obs[label])
    cell_spot = cluster_mapper_matrix @ cell_clu.T
    if thres == None:
        return cell_spot
    else:
        return (cell_spot>thres).astype(int)
    
def spot_mapping(
    scadata,
    stdata,
    cluster_mapping_matrix=None,
    x_bulk=None,
    y_bulk=None,
    genes=None,
    device = 'cpu',
    num_epochs = 500,
    learning_rate = 0.1):
    """
    Mapping the single cells to the spatial spots.
    Args:
        scadata (anndata.AnnData): scRNA-seq data
        stdata (anndata.AnnData): spatial transcriptomics data
        cluster_mapping_matrix (np.array): sc to cluster mapping matrix
        genes (list): genes to be used for mapping
        device (str): device to be used for training
        num_epochs (int): number of epochs to be used for training
        learning_rate (float): learning rate to be used for training
    Return:
        mapping_matrix (np.array): single cells to spatial spots mapping matrix
    """
    if x_bulk is None and y_bulk is None:
        x_bulk = sc.AnnData(X=generate_Xstrips(stdata),var=stdata.var)
        y_bulk = sc.AnnData(X=generate_Ystrips(stdata),var=stdata.var)
    preprocess(scadata,x_bulk,genes)
    overlap_genes = scadata.uns["overlap"]
    S = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32",)
    Gx1 = np.array(x_bulk[:, overlap_genes].X, dtype="float32")
    Gx2 = np.array(y_bulk[:, overlap_genes].X, dtype="float32")
    if stdata is None:
        mapping_matrix = cell2spots_strips(S=S,Gx=Gx1, Gy=Gx2,cluster_matrix=cluster_mapping_matrix,device=device).fit(learning_rate=learning_rate, num_epochs=num_epochs, print_each=100)
        return mapping_matrix
    ST=stdata[:,overlap_genes]
    mapping_matrix = cell2spots(S=S, ST=ST,Gx=Gx1, Gy=Gx2,cluster_matrix=cluster_mapping_matrix,device=device).fit(learning_rate=learning_rate, num_epochs=num_epochs, print_each=100)
    return mapping_matrix

def spot_mapping3D(
    scadata,
    stdata,
    cluster_mapping_matrix,
    genes=None,
    device = 'cpu',
    num_epochs = 500,
    learning_rate = 0.1):
    """
    Mapping the single cells to the spatial spots.
    Args:
        scadata (anndata.AnnData): scRNA-seq data
        stdata (anndata.AnnData): spatial transcriptomics data
        cluster_mapping_matrix (np.array): sc to cluster mapping matrix
        genes (list): genes to be used for mapping
        device (str): device to be used for training
        num_epochs (int): number of epochs to be used for training
        learning_rate (float): learning rate to be used for training
    Return:
        mapping_matrix (np.array): single cells to spatial spots mapping matrix
    """
    x_bulk = generate_Xstrips(stdata)
    y_bulk = generate_Ystrips(stdata)
    z_bulk = generate_Zstrips(stdata)
    bx1 = sc.AnnData(X=x_bulk,var=stdata.var)
    bx2 = sc.AnnData(X=y_bulk,var=stdata.var)
    bx3 = sc.AnnData(X=z_bulk,var=stdata.var)
    preprocess(scadata,bx1,genes)
    overlap_genes = scadata.uns["overlap"]
    S = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32")
    Gx1 = np.array(bx1[:, overlap_genes].X, dtype="float32")
    Gx2 = np.array(bx2[:, overlap_genes].X, dtype="float32")
    Gx3 = np.array(bx3[:, overlap_genes].X, dtype="float32")
    ST=stdata[:,overlap_genes]
    mapping_matrix = cell2spots3D(S=S, ST=ST,Gx=Gx1, Gy=Gx2,Gz=Gx3,cluster_matrix=cluster_mapping_matrix,device=device).fit(learning_rate=learning_rate, num_epochs=num_epochs, print_each=100)
    return mapping_matrix

def sc2sc(
    scadata,
    stdata_raw,
    mapping_matrix,
    sc_label,
    thres=0.5,
    method='max'):
    """
    Mapping the single cells to the single cells of every spatial spots.
    Args:
        scadata (anndata.AnnData): scRNA-seq data
        stdata_raw (anndata.AnnData): spatial transcriptomics data with single cell resolution.
        mapping_matrix (np.array): sc to spots mapping matrix
        sc_label (str): celltype label stored in the single-cell data obs
        thres (floot): thres to filter small values
        method (str): method to select the cells
    Return:
        cell_alocated_data (anndata.AnnData): scRNA-seq data with spatial information
    """
    st_x = []
    st_y = []
    select_ct = []
    select_ct_index = []
    select_gep = []
    select_cells_num = np.sum(mapping_matrix>thres,axis=0)
    raw_spot_index = stdata_raw.obs['spot_index'].unique()
    overlap_genes = scadata.uns["overlap"]
    for i in tqdm.tqdm(raw_spot_index):
        sorted_spot_indices = np.argsort(raw_spot_index)
        index_of_spot = np.where(raw_spot_index[sorted_spot_indices] == i)[0][0]
        st_temp = stdata_raw[stdata_raw.obs['spot_index']==i][:,overlap_genes]
        num_cells = select_cells_num[index_of_spot]
        if num_cells==0:
            num_cells = st_temp.shape[0]
        sc_temp = scadata[np.argsort(mapping_matrix[:,index_of_spot], axis=0)[-num_cells:],:][:,overlap_genes]
        cos_result = cos_s(sc_temp.X,st_temp.X).T
        if method=='max':
            select_index = np.argmax(cos_result, axis=1)
        elif method=='lap':
            select_index = lap.lapjv(1-cos_result,extend_cost=True)[1]
        sc_select = scadata[np.argsort(mapping_matrix[:,index_of_spot], axis=0)[-num_cells:],:][select_index]
        select_ct.extend(sc_select.obs[sc_label].values.tolist())
        select_ct_index.extend(sc_select.obs.index.tolist())
        select_gep.append(sc_select.X.toarray())
        st_x.extend(st_temp.obs.x.values.tolist())
        st_y.extend(st_temp.obs.y.values.tolist())
    cell_alocated_data = sc.AnnData(np.vstack(select_gep),obs=pd.DataFrame(select_ct,columns=[sc_label],index=select_ct_index),var=scadata.var)
    cell_alocated_data.obsm['spatial'] = np.array([st_x,st_y]).T
    return cell_alocated_data


