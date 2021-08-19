#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the GNU license.

import numpy as np
import scipy.linalg as spl
import warnings
import scipy
from scipy.sparse.linalg import lobpcg
import time
import json
    
class TimeSeriesTooShortError(Exception):
    """Raised when the time series in the input data have less than three elements"""
    pass
    

def force_symmetric(M):
    """Force the symmetry of a given matrix.

    The symmetric version is computed by first replicating the diagonal and
    upper triangular components of the matrix, and mirroring the upper diagonal
    into the lower diagonal them into the lower triangular.

    Parameters
    ----------
    M: (N, N) numpy array
       Matrix to be made symmetric

    Returns
    -------
    M_sym: Symmetric version of M
    """
    # One diag extracts the diagonal into an array, two
    # diags turns the array into a diagonal matrix.
    diag_M = np.diag(np.diag(M))
    triu_M = np.triu(M, 1)

    return triu_M + diag_M + triu_M.transpose()
    

def solve_general_eigenproblem(Q, D=None, is_symmetric=True):

    """Solve the general eigenproblem.

    Solves the general eigenproblem Qx = lambda*Dx. The eigenvectors returned are
    unitary in the norm induced by the matrix D. The eigenvalues are sorted in
    ascending order, with the eigenvectors sorted accordingly. If D is not set, 
    the identity matrix is assumed.

    Parameters
    ----------

    Q: (M, M) numpy array
       Main matrix
    D: (M, M) numpy array
       Numpy array with matrix. If not set, this function will solve the
       standard eigenproblem (Qx = lambda*x)
    is_symmetric: boolean
                  Indicates whether Q *and* D are symmetric. If set, the 
                  program will use a much faster, but less general,
                  algorithm

    Returns
    -------
    eigenvalues: numpy array with eigenvalues
    eigenvectors: numpy array with eigenvectors
                  Note: The matrix is transposed in relation to standard Numpy
    """
        
    # By default, spl.eig returns eigenvectors normalised according to the Frobenius
    # norm, while Matlab (and spl.eigh) returns them normalised by the norm induced by D.
    # We convert to the Matlab version for easier comparison.

    if is_symmetric:
        eigenvalues, eigenvectors = spl.eigh(Q, D, check_finite=False)
    else:
        eigenvalues, eigenvectors = spl.eig(Q, D, check_finite=False)
        
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    if (is_symmetric == False) and (D is not None):
        for i in range(eigenvectors.shape[1]):
            e = eigenvectors[:, i]
            n = np.matmul(e.transpose(), np.matmul(D, e))
            eigenvectors[:, i] = e/np.sqrt(n)

    # As a general recap. According to the scipy documentation,
    # A   vr[:,i] = w[i] B vr[:,i]
    # That is, the index of the eigenvector is the second index.
    # Thus, when we need to sort the eigenvectors, we only need to
    # sort in the second index.
   
    # Sort eigen pairs in increasing order of eigenvalues
    sort_eigen = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eigen]
    normalisation_factor = np.average(eigenvalues[1:])
    eigenvalues = eigenvalues/normalisation_factor
    eigenvectors = eigenvectors[:, sort_eigen]

    return eigenvalues, eigenvectors
    

def get_fiedler_eigenpair(Q, D=None, is_symmetric=True):

    """Solve the general eigenproblem to find the Fiedler vector and the corresponding eigenvalue.

    Solves the general eigenproblem Qx = lambda*Dx, and returns the Fiedler vector and associated eigenvalue. 
    The former is unitary in the norm induced by the matrix D. If D is not set, the identity matrix is assumed.

    Parameters
    ----------

    Q: (M, M) numpy array
       Main matrix
    D: (M, M) numpy array
       Matrix that accounts for node degree bias. If not set, this function will solve the
       standard eigenproblem (Qx = lambda*x)
    is_symmetric: boolean
                  Indicates whether Q *and* D are symmetric. If set, the program will use a 
                  much faster, but less general, algorithm

    Returns
    -------
    second_smallest_eigval: floating-point number
                         The second smallest eigenvalue
    fiedler_vector: (M) numpy array
                    The Fiedler vector
    """
    
    start1 = time.time()
    
    X = scipy.rand(Q.shape[0],2)
    eps = np.finfo(float).eps
    tol_standard = Q.shape[0]*np.sqrt(eps)
    tol = tol_standard*(10**(-3))
    maxiter = 50
    eigenvalues_lobpcg, eigenvectors_lobpcg, eigval_hist, res_norm_hist = lobpcg(Q, X, B=D, tol=tol, maxiter = maxiter, largest=False, retLambdaHistory=True, retResidualNormsHistory=True)
#    eigenvalues_lobpcg, eigenvectors_lobpcg, eigval_hist, res_norm_hist = lobpcg(Q, X, B=D, largest=False, retLambdaHistory=True, retResidualNormsHistory=True)
  
    print('tol_standard : ', tol_standard, '\n')

    eigenvalues_lobpcg = np.real(eigenvalues_lobpcg)
    eigenvectors_lobpcg = np.real(eigenvectors_lobpcg)
    
    sort_eigen_lobpcg = np.argsort(eigenvalues_lobpcg)
    eigenvalues_lobpcg = eigenvalues_lobpcg[sort_eigen_lobpcg]
    if D is not None:
        D_inv = np.diag(np.reciprocal(np.diag(D)))
        normalisation_factor_lobpcg = np.trace(D_inv @ Q)/(Q.shape[0]-1)
    else:
    	normalisation_factor_lobpcg = np.trace(Q)/(Q.shape[0]-1)
    second_smallest_eigval_lobpcg = eigenvalues_lobpcg[1]/normalisation_factor_lobpcg
    fiedler_vector_lobpcg = eigenvectors_lobpcg[:, sort_eigen_lobpcg[1]]
    eT_De_fiedler_lobpcg = fiedler_vector_lobpcg.transpose() @ (D @ fiedler_vector_lobpcg)
    
    end1 = time.time()
    time_taken_by_lobpcg = end1-start1
    
    
    start2 = time.time()
   
    if is_symmetric:
        eigenvalues, eigenvectors = spl.eigh(Q, D, check_finite=False)
    else:
        eigenvalues, eigenvectors = spl.eig(Q, D, check_finite=False)
        
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    sort_eigen = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_eigen]
    normalisation_factor = np.average(eigenvalues[1:])
    second_smallest_eigval = eigenvalues[1]/normalisation_factor
    
    fiedler_vector = eigenvectors[:, sort_eigen[1]]
    if (is_symmetric == False) and (D is not None):
        n = np.matmul(fiedler_vector.transpose(), np.matmul(D, fiedler_vector))
        fiedler_vector = fiedler_vector/np.sqrt(n)
        
    end2 = time.time() 
    time_taken_by_eigh = end2-start2
    

    vectorF = fiedler_vector
    vectorF_lobpcg = fiedler_vector_lobpcg
    mag_vectorF = np.sqrt(vectorF.transpose() @ vectorF)
    mag_vectorF_lobpcg = np.sqrt(vectorF_lobpcg.transpose() @ vectorF_lobpcg)
    dot_product_F = (vectorF @ vectorF_lobpcg)/(mag_vectorF*mag_vectorF_lobpcg)
    ratio_magF = mag_vectorF/mag_vectorF_lobpcg

    eigenvalue0 = eigenvalues[0]/normalisation_factor
    eigenvalue0_lobpcg = eigenvalues_lobpcg[0]/normalisation_factor_lobpcg
    eigenvalueF = second_smallest_eigval
    eigenvalueF_lobpcg = second_smallest_eigval_lobpcg

    data_for_output = {"time taken by eigh process": str(time_taken_by_eigh), "time taken by lobpcg process": str(time_taken_by_lobpcg), "dot product between eigh and lobpcg fiedler vectors": str(dot_product_F), "ratio of magnitudes of eigh and lobpcg fiedler vectors": str(ratio_magF), "smallest eigenvalue eigh": str(eigenvalue0), "algebraic connectivity eigh": str(eigenvalueF), "smallest eigenvalue lobpcg": str(eigenvalue0_lobpcg), "algebraic connectivity lobpcg": str(eigenvalueF_lobpcg), "eigenvalue history lobpcg": str(eigval_hist), "residual norms history lobpcg": str(res_norm_hist), "eT.D.e for fiedler vector lobpcg": str(eT_De_fiedler_lobpcg)}
    
    json_obj = json.dumps(data_for_output, indent = 11)
    with open("output.json", "w") as outfile:
        outfile.write(json_obj)

    return second_smallest_eigval, fiedler_vector
    

def spectral_reorder(B, method = 'geig'):
    """Computes the spectral reorder of the matrix B

    Parameters
    ----------
    B: (M, M) array_like
       Square matrix to be reordered
    method: string
            Method of reordering. Possibilities are 'geig', 'unnorm', 'rw' and 'sym'

    Returns
    -------
    sorted_B: (M, M) numpy array
              Reordered matrix
    sort_idx: (M) numpy array
              Reordering mask applied to B
    eigenvalue: floating-point number
                Second-smallest eigenvalue
    eigenvector: (M) numpy array
                 The Fiedler vector
    """

    # Fix the input
    assert B.shape[0] == B.shape[1], "Matrix B must be square!"

    min_b = np.min(B)
    assert min_b >= -1.0, "This function only accepts matrices with a mininum negative value of -1"

    if min_b < 0:
        warnings.warn("""
        The value 1 is being added to your similarity matrix to ensure positivity.
        This may cause issues with interpretation. Consider inputing a positive matrix""",
        warnings.UserWarning)
        C = B + 1
    else:
        C = B

    # Actual decomposition

    # create the laplacian matrix (Q).
    # For all non-diagonal elements, populate matrix Q with (minus) the corresponding
    # values of matrix C (i.e. Q[i,j] = -C[i,j]).
    # For each diagonal element Q[i,i], sum across the corresponding row of C (excluding the
    # diagonal element C[i,i]) and set Q[i,i] equal to that sum. 

    triuC = np.triu(C,1) # Extract upper triangular elements;
    C = triuC + triuC.transpose() # Reconstruct a symmetric weighted adjacency matrix eliminating possible small errors in off-diagonal elements
    D =  np.diag(np.sum(C, axis=-1)) # Compute the Degree matrix

    Q = D - C; # Compute un-normalised Laplacian

    method = method.lower()

    if method == 'geig':
        # Method using generalised spectral decomposition of the
        # un-normalised Laplacian (see Shi and Malik, 2000)

        eigenvalue, eigenvector = get_fiedler_eigenpair(Q, D)

    elif method == 'sym':
        # Method using the eigen decomposition of the Symmetric Normalized
        # Laplacian. Note that results should be the same as 'geig'
        T = np.sqrt(D)
        L = spl.solve(T, Q)/np.diag(T) #Compute the normalized laplacian
        L = force_symmetric(L) # Force symmetry

        eigenvalue, eigenvector = get_fiedler_eigenpair(L)
        eigenvector = spl.solve(T, eigenvector) # automatically normalized (i.e. eigenvector.transpose() @ (D @ eigenvector) = 1)

    elif method == 'rw':
        # Method using eigen decomposition of Random Walk Normalised Laplacian
        # This method has not been rigorously tested yet

        L = spl.solve(D, Q)

        eigenvalue, eigenvector = get_fiedler_eigenpair(L, is_symmetric=False)
        n = np.matmul(eigenvector.transpose(), np.matmul(D, eigenvector))
        eigenvector = eigenvector/np.sqrt(n)

    elif method == 'unnorm':

        eigenvalue, eigenvector = get_fiedler_eigenpair(Q)

    else:
        raise NameError("""Method '{}' not allowed. \n
        Please choose one of the following: 'sym', 'rw', 'geig', 'unnorm'.""".format(method))

    v2 = eigenvector # Fiedler vector
    sort_idx = np.argsort(v2) # Find the reordering index
    sorted_B = B[sort_idx,:] # Reorder the original matrix
    sorted_B = sorted_B[:,sort_idx] # Reorder the original matrix

    return sorted_B, sort_idx, eigenvalue, eigenvector


def create_affinity_matrix(neighborhood, eps=np.finfo(float).eps, verbose=False):
    """
    Computes the affinity matrix of a given neighborhood

    Parameters
    ----------
    neighborhood: (M, N) numpy array
    eps: float
         Small value which will replace negative numbers

    Returns
    -------
    affinity: (M, M) numpy array
              Affinity matrix of the neighborhood
    """
    # The following two lines are necessary for when neighborhood is an (M,) array (i.e. it is 1D), in 
    # which case it is converted to an (M,1) array. 
    neighborhood_len = neighborhood.shape[0]
    neighborhood = neighborhood.reshape(neighborhood_len, -1)
    # Here, the affinity matrix should have n_neighbors x data_size shape
    if neighborhood.shape[1] < 3:
        raise TimeSeriesTooShortError("Time series have less than 3 entries. Your analysis will be compromised!\n")

    # Create a mean centered neighborhood
    neighborhood_mean = np.mean(neighborhood, axis=-1)
    neighborhood_mc = neighborhood - neighborhood_mean.reshape(-1, 1)
    neighborhood_mc[np.abs(neighborhood_mc)<eps] = eps

    if verbose:
        print("neighborhood_mean")
        print(neighborhood_mean)

        print("neighborhood_mc")
        print(neighborhood_mc)

    # Normalise the mean centered neighborhood
    neighborhood_w = np.sqrt(np.sum(neighborhood_mc**2, axis=-1)).reshape(-1, 1)
    neighborhood_scaled = neighborhood_mc/neighborhood_w

    affinity = np.dot(neighborhood_scaled, neighborhood_scaled.transpose())
    affinity[affinity > 1.0] = 1.0
    affinity[affinity < -1.0] = -1.0


    # "Linearlise" the affinity ensure positive correlations are between 0 to 1
    # what i am doing here is to change cosines to angles but to ensure that
    # this remains a similarity rather than dissimilarity I am treating the
    # values as the sine rather than cosine. I.e. identical signals will be 90
    # rather than 0. 90/90 == 1.
    A = np.arcsin(affinity)/np.pi*180.0
    A = A/90.0
    # Remove negative correlations
    A[A<0] = eps

    if verbose:
        print("affinity")
        print(affinity)
        print("A")
        print(A)

    return A
