from .util import *
from .model import cell2clusters,cell2spots
from .ot_model import solve_OT
import lap
import pandas as pd
import numpy as np
import scanpy as sc
import tqdm
from sklearn.metrics.pairwise import cosine_similarity as cos_s
class Zmap:
    def __init__(self, 
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
                emptygrid=None,
                shape="square", 
                device='cpu'):
        """
        This class is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset(or bulk RNA datasets). 
        It will use the Zmap algorithm on the scRNA-seq dataset and reconstructe a new spatial transcriptomics data.
        The input parameters are the scRNA-seq dataset (scadata), the spatial transcriptomics dataset (stdata) or the bulk datasets (bulkX and bulkY), and the genes to be used (genes).
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
        self.cluster_matrix = None
        self.spot_matrix = None
        self.shape = shape
        self.device = device

    def allocate(self):
        """
        This function is used to run the Zmap algorithm on a scRNA-seq dataset and a spatial transcriptomics dataset. 
        It will train the Zmap algorithm on the scRNA-seq dataset and use the trained model to predict the spatial transcriptomics data.
        The output is a matrix with the same shape as the spatial transcriptomics data, where each element is a probability of the corresponding cell belonging to a specific cluster.
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
            else:
                cluster_time = self.cluster_time
                self.cluster_matrix = 0
            if cluster_time == 0:
                self.spot_matrix = spot_mapping(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)
            else:
                for i in range(cluster_time):
                    label = 'clutimes_' + str(i)
                    self.cluster_label.append(label)
                    random_cluster(self.stdata,label,n_pcs=self.n_pcs,pc_frac=self.pc_frac,samples_time=self.cluster_time,shape=self.shape,target_num = self.target_num)
                    self.cluster_matrix += cluster_mapping(self.scdata,self.stdata,self.genes,label=label,device = self.device,thres=self.cluster_thres)
                self.cluster_matrix = self.cluster_matrix/self.cluster_time
                self.spot_matrix = spot_mapping(self.scdata,self.stdata,self.cluster_matrix.values,genes=self.genes,device = self.device)
        else:
            self.scLocX = strips_mapping(self.scdata,self.bulkX,self.genes,device = self.device,thres=self.cluster_thres)
            self.scLocY = strips_mapping(self.scdata,self.bulkY,self.genes,device = self.device,thres=self.cluster_thres)
            self.spot_matrix = solve_OT(self.emptygrid,self.scLocX,self.scLocY,thres=self.cluster_thres,njob=8)




def strips_mapping(scadata,bulk,genes=None,device = 'cpu',thres=None):
    if genes is None:
        preprocess(scadata,bulk)
    else:
        preprocess(scadata,bulk,genes)
    overlap_genes = scadata.uns["overlap"]
    S = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32",)
    G = np.array(bulk[:, overlap_genes].X, dtype="float32")
    bulk_mapper = cell2clusters(S=S,G=G,device=device)
    bulk_matrix = bulk_mapper.fit(num_epochs=500,learning_rate=0.1,print_each=100)
    if thres == None:
        return bulk_matrix
    else:
        bulk_matrix[bulk_matrix<thres]=0
        return bulk_matrix

def cluster_mapping(scadata,stdata,genes=None,label='leiden',device = 'cpu',thres=None):
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
    cluster_mapper_matrix = cluster_mapper.fit(num_epochs=500,learning_rate=0.1,print_each=100)
    cell_clu = transform(stdata.obs[label])
    cell_spot = cluster_mapper_matrix @ cell_clu.T
    if thres == None:
        return cell_spot
    else:
        return (cell_spot>thres).astype(int)
    
def spot_mapping(scadata,stdata,M,genes=None,device = 'cpu'):
    x_bulk = generate_Xstrips(stdata)
    y_bulk = generate_Ystrips(stdata)
    bx1 = sc.AnnData(X=x_bulk,var=stdata.var)
    bx2 = sc.AnnData(X=y_bulk,var=stdata.var)
    preprocess(scadata,bx1,genes)
    overlap_genes = scadata.uns["overlap"]
    S = np.array(scadata[:, overlap_genes].X.toarray(), dtype="float32",)
    Gx1 = np.array(bx1[:, overlap_genes].X, dtype="float32")
    Gx2 = np.array(bx2[:, overlap_genes].X, dtype="float32")
    ST=stdata[:,overlap_genes]
    mapping_matrix = cell2spots(S=S, ST=ST,Gx=Gx1, Gy=Gx2,M=M,device=device).fit(learning_rate=0.1, num_epochs=1000, print_each=100)
    return mapping_matrix


def sc2sc(scadata,stdata_raw,mapping_matrix,thres=0.5,method='max'):
    raw_ct = []
    st_x = []
    st_y = []
    select_ct = []
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
        cos_result = cos_s(sc_temp.X,st_temp.X.A).T
        if method=='max':
            select_index = np.argmax(cos_result, axis=1)
        elif method=='lapjv':
            select_index = lap.lapjv(1-cos_result,extend_cost=True)[1]
        sc_select = scadata[np.argsort(mapping_matrix[:,index_of_spot], axis=0)[-num_cells:],:][select_index]
        select_ct.extend(sc_select.obs.subclass.values.tolist())
        select_gep.append(sc_select.X.toarray())
        raw_ct.extend(st_temp.obs.celltype.values.tolist())
        st_x.extend(st_temp.obs.x.values.tolist())
        st_y.extend(st_temp.obs.y.values.tolist())
    cell_alocated_data = sc.AnnData(np.vstack(select_gep),obs=pd.DataFrame(select_ct,columns=['subclass']),var=scadata.var)
    cell_alocated_data.obsm['spatial'] = np.array([st_x,st_y]).T
    return cell_alocated_data
