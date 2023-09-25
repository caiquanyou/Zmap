import numpy as np
import pandas as pd
import scanpy as sc
import leidenalg
import igraph as ig
import anndata
from sklearn.decomposition import PCA
from scipy.sparse import issparse
import nmslib
import random
# import SpaGCN as spg
from .custom_SpaGCN import *

def transform(label:pd.Series):
    """
    This function takes the label column from the input dataframe and converts it into a one-hot encoded dataframe.
    The function returns a dataframe with the one-hot encoded label.
    Args:
        label (pd.Series): Cell label matrix with shape (number_cells, number_label). Default is 'cell_type'.
    Returns:
        label dataframe
    """
    df = pd.get_dummies(label)
    return df

def transfer_alabel(mapping_matrix:np.ndarray, stdata:anndata.AnnData, label:pd.Series):
    """
    Transfer label from sc to st. 
    Args:
        mapping_matrix (np.ndarray): cell X spot Matrix.
        stdata (anndata.AnnData): st data object.
        label (pd.Series): Optional. Cell label matrix with shape (number_cells, number_label). Default is 'cell_type'.
    Returns:
        label dataframe
    """
    df = transform(label)
    ct_prob = pd.DataFrame(mapping_matrix.T @ df)
    ct_prob.index = stdata.obs.index
    return ct_prob

def generate_grid(stdata_raw,width):
    """this function is to generate the spatial data with spot resolution
    Args:
        stdata_raw (anndata.AnnData): spatial data with single cell resolution
        width (int): width of the spot
    Returns:
        the function will return the spot expression and update the 'x' and 'y' column in stdata.obs
    """
    stdata_raw.obs['x'] = stdata_raw.obsm['spatial'][:,0] - stdata_raw.obsm['spatial'][:,0].min()
    stdata_raw.obs['y'] = stdata_raw.obsm['spatial'][:,1] - stdata_raw.obsm['spatial'][:,1].min()
    width = width
    x_min, x_max = np.min(stdata_raw.obs['x']), np.max(stdata_raw.obs['x'])
    y_min, y_max = np.min(stdata_raw.obs['y']), np.max(stdata_raw.obs['y'])
    nx = int((x_max - x_min) // width)+1
    ny = int((y_max - y_min) // width)+1
    for i, cell in enumerate(stdata_raw):
        x, y = cell.obs['x'].values, cell.obs['y'].values
        spot_idx = int(np.floor(x/width) + np.floor(y/width) * nx)
        stdata_raw.obs.loc[cell.obs.index[0],'spot_index'] = spot_idx
    stdata_raw_df = stdata_raw.to_df()
    stdata_raw_df['spot_index'] = stdata_raw.obs['spot_index'].values
    stdata_raw_ep = stdata_raw_df.groupby('spot_index').sum()
    stdata = sc.AnnData(X=stdata_raw_ep.values,var=stdata_raw.var)
    stdata.obs['x'] = stdata_raw_ep.index.values % nx 
    stdata.obs['y'] = stdata_raw_ep.index.values // nx
    stdata.obsm['spatial'] = stdata.obs[['x','y']].values
    stdata.obs['array_col'] = stdata.obsm['spatial'][:,0]
    stdata.obs['array_col'] = stdata.obs['array_col'].astype('int32')
    stdata.obs['array_row'] = stdata.obsm['spatial'][:,1]
    stdata.obs['array_row'] = stdata.obs['array_row'].astype('int32')
    return stdata


def generate_Xstrips(stdata):
    """this function is to generate X strips, which is used for calculating the correlation matrix
    Args:
        stdata (anndata.AnnData): Visium data
    Returns:
        the function will return the X strips and update the 'x' column in stdata.obs
    """
    slides_x =  []
    x_min = stdata.obs['array_col'].min()
    x_max = stdata.obs['array_col'].max()
    lenth = x_max-x_min+1
    j = 0
    for i in range(lenth):
        idx = abs(stdata.obs['array_col'] - x_min - i) < 0.1
        if sum(idx)==0:
            continue
        slides_x.append(np.asarray(stdata.X[idx,:].sum(axis =0)).reshape(-1)) 
        stdata.obs.loc[idx,'x'] = j
        j+=1
    return np.array(slides_x)

def generate_Ystrips(stdata):
    """this function is to generate Y strips, which is used for calculating the correlation matrix
    Args:
        stdata (anndata.AnnData): Visium data
    Returns:
        the function will return the Y strips and update the 'y' column in stdata.obs
    """
    slides_y =  []
    y_min = stdata.obs['array_row'].min()
    y_max = stdata.obs['array_row'].max()
    lenth = y_max-y_min+1
    j = 0
    for i in range(lenth):
        idx = abs(stdata.obs['array_row'] - y_min - i) < 0.1
        if sum(idx)==0:
            continue
        slides_y.append(np.asarray(stdata.X[idx,:].sum(axis =0)).reshape(-1)) 
        stdata.obs.loc[idx,'y'] = j
        j+=1
    return np.array(slides_y)

def generate_cluster_exp(stdata,label):
    """Generate cluster expression
    Args:
        stdata (anndata): Spatial matrix
        label (str): clusters label stored in the sp data obs
    Returns:
        cluster expression matrix
    """
    clusters =  []
    for i in stdata.obs[label].unique():
        temp = stdata[stdata.obs[label]==i]
        clusters.append(np.asarray(temp.X.sum(axis =0)).reshape(-1)) 
    return np.array(clusters)


def preprocess(scdata, stdata ,genes=None):
    '''
    This function filters the data. It removes genes that are not expressed in any cells of the data set. it makes sure that the genes in the data set are also in X_bulk, Y_bulk, and Z_bulk.then saves the genes that are in all of these data sets in scdata.uns["overlap_genes"].
    '''
    sc.pp.filter_genes(scdata, min_cells=1)
    if genes is None:
        genes = scdata.var.index
    genes = list(set(genes) & set(scdata.var.index) & set(stdata.var.index))
    scdata.uns["overlap"] = genes


def random_cluster(stdata,label,n_pcs=50,pc_frac=0.5,samples_time=1,shape="square",target_num=10):
    """Generate random cluster label
    Args:
        stdata (anndata): Spatial matrix
        label (str): clumn name of the cluster label stored in the sp data obs
        n_pcs (int): number of PCs used for clustering
        samples_time (int): number of times to sample
    """
    pca = PCA(n_components=n_pcs)
    if issparse(stdata.X):
        pca.fit(stdata.X.A)
        embed=pca.transform(stdata.X.A)
    else:
        pca.fit(stdata.X)
        embed=pca.transform(stdata.X)
    selected_pcs_indices = random.sample(range(n_pcs), int(n_pcs*pc_frac))
    selected_pcs = embed[:, selected_pcs_indices]
    SpaGCN_cluster(selected_pcs, stdata, label, shape=shape,target_num=target_num)

def SpaGCN_cluster(selected_pcs, stdata,label,shape="square",target_num=10):
    """Generate cluster label using SpaGCN
    Args:
        selected_pcs (np.ndarray): selected PCs
        stdata (anndata): Spatial matrix
        label (str): clumn name of the cluster label stored in the sp data obs
        shape (str): shape of the spot, default is square for ST, other options is 'hex' for Visium
    Returns:
        cluster label stored in the sp data obs
    """
    prefilter_specialgenes(stdata)
    sc.pp.normalize_per_cell(stdata)
    sc.pp.log1p(stdata)
    x_pixel=stdata.obs["x"].tolist()
    y_pixel=stdata.obs["y"].tolist()
    adj_no_img=calculate_adj_matrix(x=x_pixel,y=y_pixel, histology=False)
    l=search_l(p=0.5, adj=adj_no_img, start=0.01, end=1000, tol=0.01, max_run=100)
    r_seed=t_seed=n_seed=100
    res=search_res(stdata, adj_no_img, selected_pcs,l, target_num=target_num, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
    clf=SpaGCN()
    clf.set_l(l)
    random.seed(r_seed)
    torch.manual_seed(t_seed)
    np.random.seed(n_seed)
    clf.train(
        stdata,
        adj_no_img,
        selected_pcs,
        init_spa=True,
        init="louvain",
        res=res,
        tol=5e-3,
        lr=0.05,
        max_epochs=200
    )
    y_pred, prob=clf.predict()
    stdata.obs[label]= y_pred
    stdata.obs[label]=stdata.obs[label].astype('category')
    refined_pred=refine(sample_id=stdata.obs.index.tolist(), pred=stdata.obs[label].tolist(), dis=adj_no_img, shape=shape)
    stdata.obs[label]=refined_pred
    stdata.obs[label]=stdata.obs[label].astype('category')


def fastKnn(X1, 
            X2=None, 
            n_neighbors=20, 
            metric='euclidean', 
            M=40, 
            post=0, # Buffer memory error occur when post != 0
            efConstruction=100,
            efSearch=200):
    if metric == 'euclidean':
        metric = 'l2'
    if metric == 'cosine':
        metric = 'cosinesimil'
    if metric == 'jaccard':
        metric = 'bit_jaccard'
    if metric == 'hamming':
        metric = 'bit_hamming'
    # efConstruction: improves the quality of a constructed graph but longer indexing time
    index_time_params = {'M': M,
                         'efConstruction': efConstruction, 
                         'post' : post} 
    # efSearch: improves recall at the expense of longer retrieval time
    efSearch = max(n_neighbors, efSearch)
    query_time_params = {'efSearch':efSearch}
    
    if issparse(X1):
        if '_sparse' not in metric:
            metric = f'{metric}_sparse'
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.SPARSE_VECTOR)
    else:
        index = nmslib.init(method='hnsw', space=metric, data_type=nmslib.DataType.DENSE_VECTOR)
    index.addDataPointBatch(X1)
    index.createIndex(index_time_params, print_progress=False)
    index.setQueryTimeParams(query_time_params)
    if X2 is None:
        neighbours = index.knnQueryBatch(X1, k=n_neighbors)
    else:
        neighbours = index.knnQueryBatch(X2, k=n_neighbors)
    
    distances = []
    indices = []
    for i in neighbours:
        if len(i[0]) != n_neighbors:
            vec_inds = np.zeros(n_neighbors)
            vec_dist = np.zeros(n_neighbors)
            vec_inds[:len(i[0])] = i[0]
            vec_dist[:len(i[1])] = i[1]
            indices.append(vec_inds)
            distances.append(vec_dist)        
        else:
            indices.append(i[0])
            distances.append(i[1])
    distances = np.vstack(distances)
    indices = np.vstack(indices)
    indices = indices.astype(np.int)
    if metric == 'l2':
        distances = np.sqrt(distances)
    
    return(distances, indices)
