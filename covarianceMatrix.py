# Description: This file contains the functions to compute the covariance matrix
# and the principal components of a given matrix using JAX.
# Author: Athyrson Ribeiro
# Year: 2024
# Version: 1.0

import jax.numpy as jnp

def get_jax_covariance(X1, X2):
    """Computers the covariance between two arrays

    Parameters
    ----------
    X1 : array
        Array of shape (n_samples, n_features)
    X2 : array
        Array of shape (n_samples, n_features)

    Returns
    -------
    cov : array
        Covariance between X1 and X2
    """

    X1M = X1.mean(axis=0) # mean of X1
    X2M = X2.mean(axis=0) # mean of X2
    n = X1.shape[0]       # number of samples
 
    cov = ( (X1 - X1M ) @ ( X2 - X2M ) )/n # covariance

    return cov            # return the covariance


def get_covariance_matrix(matrix_1):
    """Computes the covariance matrix of a given matrix

    Parameters
    ----------
    matrix_1 : array
        Array of shape (n_samples, n_features)
    
    Returns
    -------
    covs_array : array
        Covariance matrix of matrix_1
    """
    n_feats = matrix_1.T.shape[0]   # number of features
    covs_array = []                 # list to store the covariances
    for idx1 in range(n_feats):     # loop over the features
        f1 = matrix_1.T[idx1]       # get the feature
        f1_list = []                # list to store the covariances of f1 with all other features
        for idx2 in range(n_feats): # loop over the features
            f2 = matrix_1.T[idx2]   # get the feature
            f1_list.append(get_jax_covariance(f1,f2))  # compute the covariance and append it to the list
        covs_array.append(f1_list)  # append the list of covariances to the covs_array
    
    return jnp.array(covs_array)    # return the covs_array as a jax array

def get_principal_componentes(matrix_1, n_comps = 2):
    """
    Computes the principal components of a given matrix
    
    Parameters
    ----------
    matrix_1 : array
        Array of shape (n_samples, n_features)
    n_comps : int
        Number of principal components to return

    Returns
    -------
    transfM : array
        Array of shape (n_comps, n_samples)
    """
    cM = get_covariance_matrix(matrix_1)                # compute the covariance matrix

    eigenvalues, eigenvectors = jnp.linalg.eig(cM)  # compute the eigenvalues and eigenvectors

    indices = jnp.argsort(eigenvalues)              # sort the eigenvalues

    pcM = jnp.real(eigenvectors[indices][-n_comps:])# get the principal components

    transfM = matrix_1 @ pcM.T                      # transform the matrix

    return transfM.T                                # return the transformed matrix




