
import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport rand, RAND_MAX

DTYPE1 = np.int
ctypedef np.int_t DTYPE_t1
DTYPE2 = np.float
ctypedef np.float_t DTYPE_t2


@cython.boundscheck(False)
@cython.wraparound(False)
def gpm_init(np.ndarray[DTYPE_t2, ndim=3] IMG, int m, int K, int k_ref, int k_range):
    
    
    cdef int a = IMG.shape[0]
    cdef int b = IMG.shape[1]
    cdef int c = IMG.shape[2]
    
    cdef np.ndarray[DTYPE_t1, ndim=4] NNF= np.zeros((a,b,K,3), dtype= DTYPE1)
    cdef np.ndarray[DTYPE_t2, ndim=3] NND= np.zeros((a,b,K), dtype= DTYPE2)
        
    cdef int i, j, filled, i1, j1, k1, k2, k3, new_rand, kk, k_min, k_max, temp_i, ii, jj
    cdef float temp_f, value1, new_dist
    
    k_min= max(0, k_ref-k_range)
    k_max= min(c, k_ref+k_range+1)
    
    for i in xrange(m,a-m):
        for j in xrange(m,b-m):
            
            filled= 0
            
            while filled<K:
                
                i1= m + rand()% (a-2*m)
                j1= m + rand()% (b-2*m)
                k1= k_min + rand()% (k_max-k_min)
                
                new_rand= 1
                if i==i1 and j==j1 and k_ref== k1:
                    new_rand= 0
                else:
                    for kk in xrange(filled):
                        if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                            new_rand= 0
                            break
                        
                if new_rand==1:
                    NNF[i,j,filled,0]= i1
                    NNF[i,j,filled,1]= j1
                    NNF[i,j,filled,2]= k1
                    new_dist= 0
                    for ii in xrange(i-m,i+m+1):
                        for jj in xrange(j-m,j+m+1):
                            value1= IMG[ii,jj,k_ref]- IMG[ii-i+i1, jj-j+j1,k1]
                            new_dist+= value1*value1
                    new_dist= new_dist**0.5
                    NND[i,j,filled]= new_dist
                    filled+= 1
            
                    
            for k1 in xrange(K-1,-1,-1):
                for k2 in xrange(k1):
                    if NND[i,j,k2]>NND[i,j,k2+1]:
                        temp_f= NND[i,j,k2]
                        NND[i,j,k2]= NND[i,j,k2+1]
                        NND[i,j,k2+1]= temp_f
                        for k3 in xrange(3):
                            temp_i= NNF[i,j,k2,k3]
                            NNF[i,j,k2,k3]= NNF[i,j,k2+1,k3]
                            NNF[i,j,k2+1,k3]= temp_i

            
    return NNF, NND
    