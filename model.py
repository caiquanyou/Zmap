import numpy as np
import pandas as pd
import torch
from torch.nn.functional import softmax, cosine_similarity

# class QP_solver:
#     def __init__(
#         self,
#         scdata,
#         clusters,
#         lamb=1e-4,
#         p=None
#     ):
#         self.scdata = scdata
#         self.clusters = clusters
#         self.lamb = lamb
#         self.p = p
#     def setup_QP(self):
#         """Computing the distribution of cells on each slides by solving argmin 1/2*||Ax - b||_2 ^2 + lambda*||x||_1
#         Args:
#             X_mean (ndarray): _description_
#             lamb (float64, optional): _description_. Defaults to 1e-4.
#         """
#         A = np.array(X_mean.T, dtype = 'float64')
#         Om = sparse.csc_matrix((A.shape[0], A.shape[0]))
#         On = sparse.csc_matrix((A.shape[1], A.shape[1]))
#         Onm = sparse.csc_matrix((A.shape[1], A.shape[0]))
#         Im = sparse.eye(A.shape[0])
#         In = sparse.eye(A.shape[1])
#         P = sparse.block_diag([On, Im], format = 'csc')
#         q = np.hstack([lamb*np.ones(A.shape[1]), np.zeros(A.shape[0])])
#         Q = sparse.vstack([sparse.hstack([A, -1*Im]), sparse.hstack([In, Onm])], format='csc')
#         b = np.zeros(A.shape[0])
#         l = np.hstack([b, np.zeros(A.shape[1])]) 
#         u = np.hstack([b, np.inf * np.ones(A.shape[1])])
#         #Create an OSQP solver
#         prob = osqp.OSQP()
#         prob.setup(P, q, Q, l, u, rho = 100, eps_prim_inf = 1e-4, eps_dual_inf = 1e-4, eps_abs = 1e-3, eps_rel = 1e-3, max_iter = 1000,verbose=False)
#         return prob
#     def solve_QP(self):
#         """Computing the distribution of cells on each slides by solving argmin 1/2*||Ax - b||_2 ^2 + lambda*||x||_1
#         """
#         global A,BXn,BYn,prob
#         A = np.array(X_mean.T, dtype = 'float64')
#         BXn = X
#         BYn = Y
#         prob = p
#         print('Starting calculate probability of cells in X strips')
#         for k in range(BXn.shape[1]):
#             b = BXn[:,k]                      
#             l_new = np.hstack([b, np.zeros(A.shape[1])]) 
#             u_new = np.hstack([b, np.inf * np.ones(A.shape[1])])
#             prob.update(l=l_new, u=u_new)
#             res = prob.solve()
#             scLocX[:,k] = res.x[:A.shape[1]]

#         return np.array(scLocX).T
    


class cell2clusters:
    """
    This class implements an algorithm that maps each cell to its cluster.
    The algorithm works as follows:
    * Create clusters with spatial expression matrix.
    * Mapping the cells to the clusters that the reconstructed cells expression vector is most similar to the raw spatial expression matrix.
    * Return the mapping matrix.
    """
    def __init__(
        self,
        scdata,
        clusters,
        lambda_g1=1.0,
        lambda_g2=1.0,
        device="cpu",
        random_state=None,
    ):
        """
            scdata (ndarray): single cell expression matrix. 
            clusters (ndarray): spatial cluster expression matrix.  
            lambda_g1 (float): weight for the first term in the loss function. Default is 1.0.
            lambda_g2 (float): weight for the second term in the loss function. Default is 1.0.
            device (str): Optional. Device to use. Default is 'cpu'.
            random_state (int): Optional. Random state. Default is None.         
        """
        self.scdata = torch.tensor(scdata, device=device, dtype=torch.float32)
        self.clusters = torch.tensor(clusters, device=device, dtype=torch.float32)
        self.lambda_g1 = lambda_g1
        self.lambda_g2 = lambda_g2
        self.random_state = random_state
        self.cluster_matrix = torch.tensor(
            np.random.normal(0, 1, (scdata.shape[0], clusters.shape[0])), device=device, requires_grad=True, dtype=torch.float32
        )

    def _loss_fn(self,verbose=False):
        """
        cell2cluster loss function.
        """
        cluster_probs = softmax(self.cluster_matrix, dim=1)
        G_rec = torch.matmul(cluster_probs.t(), self.scdata)
        cluster_cos1 = self.lambda_g1 * cosine_similarity(G_rec, self.clusters, dim=0).mean()
        cluster_cos2 = self.lambda_g2 * cosine_similarity(G_rec, self.clusters, dim=1).mean()
        expression_term = cluster_cos1 + cluster_cos2
        total_loss = -expression_term
        if verbose:
            print("total_loss: {:.3f}".format(total_loss.tolist()))
        return total_loss, cluster_cos1, cluster_cos2
    def fit(self, num_epochs=500, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.
        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
        Returns:
            cluster_probs (ndarray): is the optimized mapping matrix, shape = (number_cells, number_clusters).
        """

        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.cluster_matrix], lr=learning_rate)
        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)
            loss = run_loss[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            cluster_probs = softmax(self.cluster_matrix, dim=1).cpu().numpy()
            return cluster_probs

class cell2spots:
    """
    This class implements an algorithm that maps each cell to its spot.
    The algorithm works as follows:
    * Create spots with spatial expression matrix.
    * Mapping the cells to the spots that the reconstructed cells expression vector is most similar to the raw spatial expression matrix.
    * Return the mapping matrix.
    """
    def __init__(
        self,
        S,
        ST,
        Gx,
        Gy,
        Gz=None,
        lambda_gx1=1,
        lambda_gx2=0,
        lambda_gy1=1,
        lambda_gy2=0,
        lambda_gz1=1,
        lambda_gz2=0,
        cluster_matrix=None,
        device="cpu",
        random_state=None,
    ):
        
        self.S = torch.tensor(S, device=device, dtype=torch.float32)
        self.ST = ST
        self.Gx = torch.tensor(Gx, device=device, dtype=torch.float32)
        self.Gy = torch.tensor(Gy, device=device, dtype=torch.float32)
        self.lambda_gx1 = lambda_gx1
        self.lambda_gx2 = lambda_gx2
        self.lambda_gy1 = lambda_gy1
        self.lambda_gy2 = lambda_gy2
        self.lambda_gz1 = lambda_gz1
        self.lambda_gz2 = lambda_gz2
        self.random_state = random_state
        if cluster_matrix is None:
            self.cluster_matrix = np.random.normal(0, 1, (S.shape[0], ST.shape[0]))
            self.cluster_matrix = torch.tensor(
                self.cluster_matrix, device=device, requires_grad=True, dtype=torch.float32
            )
        else:
            self.cluster_matrix = cluster_matrix
            self.cluster_matrix = torch.tensor(
                self.cluster_matrix, device=device, requires_grad=True, dtype=torch.float32
            )
            # self.rawM = self.M.clone().detach() > 0
            self.mask = self.cluster_matrix.clone().detach() < 1
        self.device = device
        self.x_length = np.int32(self.ST.obs['x'].max() - self.ST.obs['x'].min() +1)
        self.y_length = np.int32(self.ST.obs['y'].max() - self.ST.obs['y'].min() +1)
        
        self.x_index = self.ST.obs['x'] - self.ST.obs['x'].min()
        self.x_index = torch.tensor(
            self.x_index, device=device, requires_grad=False, dtype=torch.int32
        )
        self.y_index = self.ST.obs['y'] - self.ST.obs['y'].min()
        self.y_index = torch.tensor(
            self.y_index, device=device, requires_grad=False, dtype=torch.int32
        )
        if Gz is not None:
            self.Gz = torch.tensor(Gz, device=device, dtype=torch.float32)
            self.z_length = np.int32(self.ST.obs['z'].max() - self.ST.obs['z'].min() +1)
            self.z_index = self.ST.obs['z'] - self.ST.obs['z'].min()
            self.z_index = torch.tensor(
                self.z_index, device=device, requires_grad=False, dtype=torch.int32
            )
        else:
            self.Gz = Gz
        
    def _generate_Xstrips(self, cluster_probs):
        """
        Generate X matrix
        """
        mapping_x = torch.zeros((self.x_length, self.S.shape[0]), device=self.device)
        mapping_x.index_add_(0, self.x_index, cluster_probs.T)
        return mapping_x

    def _generate_Ystrips(self, cluster_probs):
        """
        Generate Y matrix
        """
        mapping_y = torch.zeros((self.y_length, self.S.shape[0]), device=self.device)
        mapping_y.index_add_(0, self.y_index, M_probs.T)
        return mapping_y
    
    def _generate_Zstrips(self, cluster_probs):
        """
        Generate Z matrix
        """
        mapping_z = torch.zeros((self.z_length, self.S.shape[0]), device=self.device)
        mapping_z.index_add_(0, self.z_index, cluster_probs.T)
        return mapping_z

    def _loss_fn(self, verbose=True):
        """
        cell2spot loss function.
        """
        cluster_probs = softmax(self.cluster_matrix, dim=1)
        mask_sum = torch.masked_select(cluster_probs,self.mask).mean() 
        Mx = self._generate_Xstrips(cluster_probs)
        My = self._generate_Ystrips(cluster_probs)
        dx_pred = torch.log(
            Mx.T.sum(axis=0) / Mx.shape[1]
        )  # KL wants the log in first argument
        dy_pred = torch.log(
            My.T.sum(axis=0) / My.shape[1]
        )
        Gx_pred = torch.matmul(Mx, self.S)
        Gy_pred = torch.matmul(My, self.S)
        gx1 = self.lambda_gx1 * cosine_similarity(Gx_pred, self.Gx, dim=0).mean()
        gx2 = self.lambda_gx2 * cosine_similarity(Gx_pred, self.Gx, dim=1).mean()
        gy1 = self.lambda_gy1 * cosine_similarity(Gy_pred, self.Gy, dim=0).mean()
        gy2 = self.lambda_gy2 * cosine_similarity(Gy_pred, self.Gy, dim=1).mean()
        cos_x = gx1 + gx2
        cos_y = gy1 + gy2
        main_x_loss1 = (gx1 / self.lambda_gx1).tolist()
        main_y_loss1 = (gy1 / self.lambda_gy1).tolist()
        main_x_loss2 = (gx2 / self.lambda_gx2).tolist()
        main_y_loss2 = (gy2 / self.lambda_gy2).tolist()
        total_loss = -cos_x - cos_y  + mask_sum
        if self.Gz is not None:
            Mz = self._generate_Zstrips(cluster_probs)
            Gz_pred = torch.matmul(Mz, self.S)
            gz1 = self.lambda_gz1 * cosine_similarity(Gz_pred, self.Gz, dim=0).mean()
            gz2 = self.lambda_gz2 * cosine_similarity(Gz_pred, self.Gz, dim=1).mean()
            main_z_loss1 = (gz1 / self.lambda_gz1).tolist()
            main_z_loss2 = (gz2 / self.lambda_gz2).tolist()
            cos_z = gz1 + gz2
            total_loss = total_loss -cos_z 
            if verbose:
                print("total_loss: {:.3f}".format(total_loss.tolist()))
            return (
                total_loss,
                main_x_loss1,
                main_y_loss1,
                main_z_loss1,
                mask_sum)
        
        if verbose:
            print("total_loss: {:.3f}".format(total_loss.tolist()))
        return (
            total_loss,
            main_x_loss1,
            main_y_loss1,
            mask_sum)

    def fit(self, num_epochs, learning_rate=0.1, print_each=100):
        """
        Run the optimizer and returns the mapping outcome.
        Args:
            num_epochs (int): Number of epochs.
            learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
            print_each (int): Optional. Prints the loss each print_each epochs. If None, the loss is never printed. Default is 100.
        Returns:
            M (ndarray): is the optimized mapping matrix, shape = (number_cells, number_spots).
        """
        if self.random_state:
            torch.manual_seed(seed=self.random_state)
        optimizer = torch.optim.Adam([self.cluster_matrix], lr=learning_rate)
        for t in range(num_epochs):
            if print_each is None or t % print_each != 0:
                run_loss = self._loss_fn(verbose=False)
            else:
                run_loss = self._loss_fn(verbose=True)
            loss = run_loss[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            spot_matrix = softmax(self.cluster_matrix, dim=1).cpu().numpy()
            return spot_matrix
