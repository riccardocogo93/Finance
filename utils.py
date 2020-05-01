from scipy.optimize import minimize
from sklearn.neighbors.kde import KernelDensity
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

### FOR PORTFOLIO OPTIMIZATION, WE USE COVARIANCE MATRIXES (SIMULATED) AND WE COMPARE THEM TO THEORETICAL ONE ###
### SO I TRY TO DENOISE EACH EMPIRICAL COVARIANCE MATRIX AND SEE WHAT HAPPENS ###

### UTILS ###

def corrToCov(lst_corr, lst_std):
    """
    Function that derives the covariance matrix from a correlation matrix
    :param lst_corr: correlation matrix
    :param lst_std: list of standard deviations
    return: lst_cov: covariance matrix
    """
    lst_cov = lst_corr*np.outer(lst_std, lst_std)
    return lst_cov

def covToCorr(lst_cov):
    """
    Function that derives the correlation matrix from a covariance matrix
    :param lst_cov: covariance matrix
    return: lst_corr: correlation matrix
    """
    # Extract the diagonal of the covariance matrix
    lst_std = np.sqrt(np.diag(lst_cov))
    
    # Compute the correlation matrix
    lst_corr = lst_cov/np.outer(a=lst_std,
                                b=lst_std)
    
    # Fix numerical errors
    lst_corr[lst_corr < -1] = -1
    lst_corr[lst_corr > 1 ] = 1
    return lst_corr

def get_valvec(lst_matrix):
    """
    Function that gets eigenvalues and eigenvectors from a Hermitian matrix
    :param lst_matrix: Hermitian matrix
    return: lst_eVals, lst_eVecs: lists of eigenvalues (ordered de-ascending and "diagonalized") and eigenvectors of lst_matrix
    """
    # Get eigenvalues and eigenvectors
    lst_eVals, lst_eVecs = np.linalg.eig(a=lst_matrix)
    
    # Sort the eigenvalues in descending order
    lst_indices = np.argsort(a=lst_eVals)[::-1]
    lst_eVals, lst_eVecs = lst_eVals[lst_indices], lst_eVecs[:, lst_indices]
    
    # Make the list of (sorted) eigenvalues the diagonal of a two-dimensional matrix
    lst_eVals = np.diagflat(v=lst_eVals)
    return lst_eVals, lst_eVecs



### FORM TRUE (REALISTIC) COVARIANCE MATRIX ###

def formBlockMatrix(int_nBlocks, int_bSize, flt_bCorr):
    """
    Function that creates a REAL (realistic) correlation matrix out of given number of blocks with a given correlation
    :param int_nBlocks: number of blocks
    :param int_bSize: size of each block
    :param flt_bCorr: correlation of each block
    return: lst_corr: correlation matrix
    """
    # Create the single block, i.e. a int_bSizexint_bSize matrix formed by the value flt_bCorr only
    lst_block = np.ones((int_bSize, int_bSize))*flt_bCorr

    # The diagonal of the block is made up of only 1 (of course, being a correlation matrix)
    lst_block[range(int_bSize), range(int_bSize)] = 1

    # Create a matrix made up of int_nBlocks of these blocks along the diagonal: this is the correlation matrix
    lst_corr = block_diag(*([lst_block]*int_nBlocks))
    return lst_corr

def formTrueMatrix(int_nBlocks, int_bSize, flt_bCorr, flt_std_low=0.05, flt_std_high=0.2):
    """
    Function that forms a REAL (realistic) covariance matrix along with its vector of means.
    The covariance matrix is made out of a given correlation matrix:
    (a) out of a given number of blocks
    (b) each block has a given size
    (c) each offdiagonal elements within each block have a given correlation
    This covariance matrix is a  representation of a TRUE (nonempirical) detoned correlation matrix of S&P500.
    In fact, each block is associated with an economic sector.
    Assumptions (without loss of generality):
    (1) The variances are drawn from a uniform distribution bounded between given limits
    (2) The vector of means is drawn from a Normal distribution with both mean and std equal to the std from the covariance matrix
    This is consistent with the notion that in an efficient market all securities have the same expected Sharpe ratio
    :param int_nBlocks: number of blocks
    :param int_bSize: size of each block
    :param flt_bCorr: correlation of each block
    :param flt_std_low: lower bound for standard deviation sampling (default: 5%)
    :param flt_std_high: upper bound for standard deviation sampling (default: 20%)
    return: lst_mu, lst_cov: vector of means and covariance matrix
    """
    # Create block correlation matrix
    lst_corr = formBlockMatrix(int_nBlocks=int_nBlocks,
                               int_bSize=int_bSize,
                               flt_bCorr=flt_bCorr)
    
    # Shuffle the columns of the correlation matrix
    dtf_corr = pd.DataFrame(data=lst_corr)
    lst_cols = dtf_corr.columns.tolist()
    np.random.shuffle(lst_cols)
    lst_corr = dtf_corr[lst_cols].loc[lst_cols].copy(deep=True).to_numpy()
    
    # Sample a list of standard deviations for covariance matrix
    lst_std = np.random.uniform(low=flt_std_low,
                                high=flt_std_high,
                                size=lst_corr.shape[0])
    
    # Convert correlation matrix into covariance matrix
    lst_cov = corrToCov(lst_corr=lst_corr,
                        lst_std=lst_std)
    
    # Sample a list of means
    lst_mu = np.random.normal(loc=lst_std,
                              scale=lst_std,
                              size=lst_cov.shape[0]).reshape(-1,1)
    
    return lst_mu, lst_cov



### FORM EMPIRICAL COVARIANCE MATRIX OUT OF REALISTIC (TRUE) COVARIANCE MATRIX###

def simulate_covMu(lst_mu, lst_cov, int_T, bln_shrink=False):
    """
    Function that uses the true (NONEMPIRICAL) covariance matrix to draw a random matrix X of size TxN.
    By calculating the mean and covariance of this matrix X, the associated EMPIRICAL covariance matrix and vector of means are derivated
    :param lst_mu: list of means of the true (nonempirical) covariance matrix
    :param lst_cov: true (nonempirical) covariance matrix
    :param int_T: number of random variables in X (i.e., number of observations)
    :param bln_shrink: whether or not to performs a Ledoit–Wolf shrinkage of the empirical covariance matrix
    return: lst_mu_emp, lst_cov_emp: empirical list of means and covariance matrix
    """
    # Compute the multivariate random variable out of the true (nonempirical) parameters
    lst_X = np.random.multivariate_normal(mean=lst_mu.flatten(),
                                          cov=lst_cov,
                                          size=int_T)
    
    # Extract the empirical means and covariance (shrikend or not) matrix
    lst_mu_emp = lst_X.mean(axis=0).reshape(-1,1)
    if bln_shrink:
        lst_cov_emp = LedoitWolf().fit(lst_X).covariance_
        
    else:
        lst_cov_emp = np.cov(m=lst_X,
        					 rowvar=0)
        
    return lst_mu_emp, lst_cov_emp



### DENOISE THE EMPIRICAL COVARIANCE MATRIX ###

def mp_pdf(flt_var, flt_q, int_num_eVals):
    """
    Function that calculates the theoretical Marcenko-Pastur probability density function.
    Consider a matrix X of i.i.d. random observations with
    (1) size of X = TxN
    (2) mean of the underlying process generating the observations = 0
    (3) variance of the underlying process generating the observations = flt_var
    :param flt_var: variance of the underlying process generating the observations
    :param flt_q: T/N
    :param int_num_eVals: number of simulated eigenvalues
    return: dtf_pdf: Marcenko-Pastur pdf evaluated on simulated eigenvalues
    """
    # Consider the matrix C = X'X/T
    # The eigenvalue of C asymptotically converge to the MP pdf (as N,T go to infinity and 1 < q < infinity)
    
    # Calculate minimum and maximum expected eigenvalues
    flt_lambda_min = flt_var*np.square(1-np.sqrt(1/flt_q))
    flt_lambda_max = flt_var*np.square(1+np.sqrt(1/flt_q))
    
    # So the eigenvalues CONSISTENT WITH RANDOM BEHAVIOR are between these two values
    # MP pdf is equal to zero out of this range
    # The eigenvalues in [0, flt_lambda_max] are associated with noise
    lst_eVals = np.linspace(start=flt_lambda_min,
                            stop=flt_lambda_max,
                            num=int_num_eVals)
    
    # Calculate MP pdf in the range where it's not equal to zero
    lst_pdf = flt_q*np.sqrt((flt_lambda_max-lst_eVals)*(lst_eVals-flt_lambda_min))/(2*np.pi*flt_var*lst_eVals)
    
    # Return MP pdf as pandas series
    dtf_pdf = pd.Series(data=lst_pdf,
                        index=lst_eVals)
    
    return dtf_pdf

def fitKDE(lst_obs, flt_bandwidth=0.25, str_kernel="gaussian", lst_x_eval=None):
    """
    Function that fits a (given) Kernel Density Estimator to a series of observations, and derive the probability of observation
    :param lst_obs: the list of observations
    :param flt_bandwidth: the bandwidth of the kernel
    :param str_kernel: the kernel to use
    :param lst_x_eval: array of values on which the fitted KDE will be evaluated
    return: dtf_pdf: empirical pdf
    """
    # List of observations lst_obs must be a 2-dimensional array
    if len(lst_obs.shape) == 1:
        lst_obs = lst_obs.reshape(-1, 1)
    
    # Initialize the KDE and fit it on the observations
    skl_kde = KernelDensity(bandwidth=flt_bandwidth,
                            kernel=str_kernel).fit(lst_obs)
    
    # List lst_x_eval must be a 2-dimensional array too
    # If lst_x_eval it's not provided, let's initialize it as the list of unique observations
    if lst_x_eval is None:
        lst_x_eval = np.unique(lst_obs).reshape(-1, 1)
        
    if len(lst_x_eval.shape) == 1:
        lst_x_eval = lst_x_eval.reshape(-1, 1)
    
    # Evaluate the log density model on the data (i.e., on lst_x_eval)
    lst_logProb = skl_kde.score_samples(X=lst_x_eval)
    
    # Return the evaluations as pandas series
    dtf_pdf = pd.Series(data=np.exp(lst_logProb),
                        index=lst_x_eval.flatten())
    
    return dtf_pdf

def error_PDFs(flt_var, lst_eVals, flt_q, flt_bandwidth, int_num_eVals=1000):
    """
    Function that calculates the square error between Theoretical MP pdf and Empirical pdf.
    This function will be used to calculate the variance that minimizes this difference.
	To calculate the Theoretical MP pdf you need:
	(1) The variance,flt_var
	(2) The ratio between the sizes of the matrix of the processes, flt_q
	(3) The number of simulated observations (eigenvalues), int_num_eVals
	To calculate the Empirical pdf you need:
	(1) The empirical eigenvalues, lst_eVals
	(2) The bandwidth of the kernel used to fit the theoretical kernel, flt_bandwidth
	(3) The points where the fitted kernel will be evaluated, i.e. the same points where the Theoretical MP pdf is evaluated
    :param flt_var: variance of the underlying process generating the observations
    :param lst_eVals: list of observations (diagonalized eigenvalues of the empirical correlation matrix)
    :param flt_q: T/N (referred to the underlying process)
    :param flt_bandwidth: the bandwidth of the kernel
    :param int_num_eVals: number of simulated observations (eigenvalues)
    return: flt_se: square error between empirical and theoretical pdfs
    """
    # Theoretical MP pdf
    dtf_pdf_theo = mp_pdf(flt_var=flt_var[0],
                          flt_q=flt_q,
                          int_num_eVals=int_num_eVals)
    
    # Empirical pdf, evaluated on the same points of the theoretical pdf
    # (i.e., dtf_pdf_theo.index.values, i.e. int_num_eVals)
    dtf_pdf_emp = fitKDE(lst_obs=lst_eVals,
                         flt_bandwidth=flt_bandwidth,
                         lst_x_eval=dtf_pdf_theo.index.values)
    
    
    # Return square error
    flt_se = np.sum(np.square(dtf_pdf_emp-dtf_pdf_theo))
    return flt_se

def find_max_eVal(lst_eVals, flt_q, flt_bandwidth):
    """
    Function that finds the maximum random (empirical) eigenvalue by fitting Marcenko’s dist.
    What does "Fitting Marcenko's dist" means?
    The idea is to minimize the square error between the empirical and theoretical pdf's as FUNCTION OF VARIANCE
    :param lst_eVals: list of observations (diagonalized eigenvalues of correlation matrix)
    :param flt_q: T/N (referred to the underlying process)
    :param flt_bandwidth: the bandwidth of the kernel
    return: flt_max_eVal, flt_var: maximum empirical eigenvalue and corresponding variance
    """
    # Find variance that minimizes the square error between the pdf's
    skl_out = minimize(fun=lambda *x: error_PDFs(*x),
                       x0=0.5,
                       args=(lst_eVals, flt_q, flt_bandwidth),
                       bounds=((1e-5, 1-1e-5),))
    
    if skl_out.success:
        flt_var = skl_out.x[0]
        
    else:
        flt_var = 1
    
    # Find maximum eigenvalue (lambda_max of MP pdf)
    flt_max_eVal = flt_var*np.square(1+np.sqrt(1/flt_q))
    return flt_max_eVal, flt_var

def denoiseCorr_CRE(lst_eVals, lst_eVecs, int_nFacts):
    """
    Function that denoise an empirical correlation matrix via Constant Residual Eigenvalue Method (i.e., by fixing random eigenvalues).
    Compared to a shrinkage method, this one removes noise while preserving the signal
    :param lst_eVals: list of observations (list of eigenvalues of correlation matrix)
    :param lst_eVecs: eigenvectors of correlation matrix
    :param int_nFacts: number of observation x random variable, i.e. number of random variables with some signal
    return: lst_corr_den: denoised correlation matrix
    """
    # Put list of eigenvales in the diagonal of a null matrix, as usual
    lst_eVals_d = np.diag(lst_eVals).copy()
    
    # Apply CRE method
    lst_eVals_d[int_nFacts:] = lst_eVals_d[int_nFacts:].sum()/float(lst_eVals_d.shape[0]-int_nFacts)
    lst_eVals_d = np.diag(lst_eVals_d)
    
    # Compute covariance matrix
    lst_cov_den = np.dot(lst_eVecs, lst_eVals_d).dot(lst_eVecs.T)
    
    # Convert it into correlation matrix
    lst_corr_den = covToCorr(lst_cov=lst_cov_den)
    return lst_corr_den

def denoiseCov_CRE(lst_cov, flt_q, flt_bandwidth):
    """
    Function that denoise an empirical covariance matrix via Constant Residual Eigenvalue Method
    :param lst_cov: empirical covariance matrix
    :param flt_q: T/N (referred to the underlying process)
    :param flt_bandwidth: the bandwidth of the kernel for the fitting-on-empitical part
    return: lst_cov_den: denoised covariance matrix
    """
    # 1. Switch from the covariance matrix to the correlation matrix
    # 2. Denoise the correlation matrix
    # 3. Return to the covariance matrix

    # Calculate correlation matrix out of covariance matrix
    lst_corr_emp = covToCorr(lst_cov=lst_cov)
    
    # To denoise the correlation matrix, you need:
    # (1) Its eigenvalues and eigenvectors
    # (2) Number of random variables with some signal
    # Extract empirical eigenvalues and eigenvectors of empirical correlation matrix
    lst_eVals_emp, lst_eVecs_emp = get_valvec(lst_matrix=lst_corr_emp)
    
    # Calculate the empirical variance
    flt_max_eVal_emp, flt_var_emp = find_max_eVal(lst_eVals=np.diag(lst_eVals_emp),
                                                  flt_q=flt_q,
                                                  flt_bandwidth=flt_bandwidth)
    
    # Sort the eigenvalues and find the index of the biggest one
    int_max_eVal_emp_index = np.diag(lst_eVals_emp)[::-1].searchsorted(flt_max_eVal_emp)

    # Number of random variables with signal, i.e. number of eigenvalues untill the biggest one
    int_nFacts = lst_eVals_emp.shape[0]-int_max_eVal_emp_index
    
    # Denoise the correlation matrix
    lst_corr_den = denoiseCorr_CRE(lst_eVals=lst_eVals_emp,
                                   lst_eVecs=lst_eVecs_emp,
                                   int_nFacts=int_nFacts)
    
    # Convert it to the covariance matrix and return it
    lst_cov_den = corrToCov(lst_corr=lst_corr_den,
                            lst_std=np.sqrt(np.diag(lst_cov)))
    
    return lst_cov_den



### CLUSTERING ###

def clusterKMeansBase(lst_corr, int_max_numClusters=10, int_num_init=10):
    """
    Function that implements the Base Clustering Algorithm
    """
    lst_x = np.sqrt((1-pd.DataFrame(data=lst_corr).fillna(0))/2)
    
    # Init the observations matrix
    dtf_silh = pd.Series()
    for int_n in range(int_num_init):
        for int_i in range(2, int_max_numClusters+1):
            skl_kMeans_ = KMeans(n_clusters=int_i,
                                 n_jobs=1,
                                 n_init=1)
            
            skl_kMeans_ = skl_kMeans_.fit(lst_x)
            lst_silh = silhouette_samples(X=lst_x,
                                          labels=skl_kMeans_.labels_)
            
            lst_stats = (lst_silh.mean()/lst_silh.std(), dtf_silh.mean()/dtf_silh.std())
            if np.isnan(lst_stats[1]) or lst_stats[0] > lst_stats[1]:
                dtf_silh = lst_silh
                skl_kMeans = skl_kMeans_
            
    lst_newId = np.argsort(skl_kMeans.labels_)
    
    # Re-order rows of correlation matrix
    dtf_corr = pd.DataFrame(data=lst_corr)
    dtf_corr_ord = dtf_corr.iloc[lst_newId]
    
    # Re-order columns of correlation matrix
    dtf_corr_ord = dtf_corr_ord.iloc[:, lst_newId]
    
    # Set the member of the clusters
    dct_clusters = {int_i: dtf_corr.columns[np.where(skl_kMeans.labels_==int_i)[0]].tolist()
                   for int_i in np.unique(skl_kMeans.labels_)}
    
    dtf_silh = pd.Series(data=lst_silh,
                         index=lst_x.index)
    
    return dtf_corr_ord.to_numpy(), dct_clusters, dtf_silh



### ALLOCATION ###

def minVar_port(lst_cov, lst_mu=None):
    """
    Function that calculates the minimum variance portfolio (if lst_mu=ones) or maximum Sharpe ratio out of an empirical covariance matrix
    :param lst_cov: empirical covariance matrix
    :param lst_mu: list of empirical means (default: None)
    return: lst_w: list of weights of the portfolio
    """
    lst_cov_inv = np.linalg.inv(lst_cov)
    lst_ones = np.ones(shape=(lst_cov_inv.shape[0], 1))
    if lst_mu is None:
        lst_mu = lst_ones
        
    lst_w = np.dot(lst_cov_inv, lst_mu)
    lst_w /= np.dot(lst_ones.T, lst_w)
    return lst_w

def optPort_nco(lst_cov, lst_mu=None, int_max_numClusters=None):
	"""
	Function that calculates the allocation via Nested Clustered Optimization Algorithm
	:param lst_cov: (de-noised) empirical covariance matrix
	:param lst_mu: empirical means
	:param int_max_numClusters: maximum number of clusters
	return: dtf_nco_alloc: dataframe with the allocation weights
	"""
	dtf_cov = pd.DataFrame(data=lst_cov)
	if lst_mu is not None:
		dtf_mu = pd.Series(data=lst_mu[:, 0])

    ### STEP 1: CLUSTER THE CORRELATION MATRIX ###
    
    # Convert covariance matrix to correlation matrix
	lst_corr = covToCorr(lst_cov=lst_cov)

    # Cluster the correlation matrix
	lst_corr, dct_clusters, _ = clusterKMeansBase(lst_corr=lst_corr,
												  int_max_numClusters=int_max_numClusters,
												  int_num_init=10)
    
    ### STEP 2: INTRACLUSTER WEIGHTS ###

    # Use the denoised covariance matrix
	dtf_wIntra = pd.DataFrame(data=0,
							  index=dtf_cov.index,
							  columns=dct_clusters.keys())
    
    # For every cluster, find the optimal allocation using the minVar_port function
	for int_i in dct_clusters:
		lst_cov_ = dtf_cov.loc[dct_clusters[int_i], dct_clusters[int_i]].values
		if lst_mu is None:
			lst_mu_ = None
            
		else:
			lst_mu_ = lst_mu.loc[dct_clusters[int_i]].values.reshape(-1,1)
    
		dtf_wIntra.loc[dct_clusters[int_i], int_i] = minVar_port(lst_cov=lst_cov_,
																 lst_mu=lst_mu_).flatten()
    
	### STEP 3: INTERCLUSTER WIEGTHS ###

	# Compute the REDUCED COVARIANCE MATRIX, that reports the correlations between clusters
	# So, the clustering and intracluster optimization steps allow to transform a the Mrkowitz problem
	# into a well-behaved one, with rho close to 0
	dtf_cov_ = dtf_wIntra.T.dot(np.dot(dtf_cov,dtf_wIntra))
	lst_mu_ = (None if lst_mu is None else dtf_wIntra.T.dot(lst_mu))
	dtf_wInter = pd.Series(data=minVar_port(lst_cov=dtf_cov_.to_numpy(),
										    lst_mu=lst_mu_).flatten(),
                           
						   index=dtf_cov_.index)
    
    # Final allocation
	dtf_nco_alloc = dtf_wIntra.mul(dtf_wInter,
								   axis=1).sum(axis=1).values.reshape(-1,1)
    
	return dtf_nco_alloc