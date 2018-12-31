# cython: profile=True

#import numpy as np
cimport numpy as np
import cython
from libc.stdlib cimport rand, RAND_MAX
from cython.parallel import parallel, prange

#DTYPE1 = np.int
#ctypedef np.int_t DTYPE_t1
#DTYPE2 = np.float
#ctypedef np.float_t DTYPE_t2


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gpm_iter(float[:,:,::1] IMG, float[:,:,::1] IMG2, int[:,:,:,::1] NNF, float[:,:,::1] NND, int[:,::1] search_mask, 
             int max_iter, int m, int K, int k_ref, int k_range, float sim_threshold, 
             int n_rand_search, int i_rad, int w_loc_i):
    
    
    cdef int a = IMG.shape[0]
    cdef int b = IMG.shape[1]
    cdef int c = IMG.shape[2]

#    cdef int n_rand_search= 10
#    cdef int i_rad= 32
    cdef int j_rad= i_rad
    cdef int k_rad= i_rad/2
    cdef int i_rad_2= i_rad/2
    cdef int j_rad_2= j_rad/2
    cdef int k_rad_2= k_rad/2
    
        
#    cdef np.ndarray[DTYPE_t1, ndim=1] new_indx= np.zeros(K, dtype= DTYPE1)
    
#    cdef float upper_x = RAND_MAX/(a- 2.0*m-1)
#    cdef float upper_y = RAND_MAX/(b- 2.0*m-1)
    
    cdef int i, j, k, l, i1, j1, k1, new_rand, kk, z, i0, j0, k0, ii, jj, new_indx, k_min, k_max, j11, i11, iw, written
    cdef int i_mn, i_mx,j_mn, j_mx,k_mn, k_mx #, i_rad_2, j_rad_2, k_rad_2
    cdef float new_dist, value1 #, dist_mean
    
#    cdef float dist_threshold= 0.0
#    cdef float dist_threshold_divisor= 1.0/( (a-2.0*m)*(b-2.0*m)*K )
    
    k_min= max(0, k_ref-k_range)
    k_max= min(c, k_ref+k_range+1)
    
#    cdef int w_loc_i= 5
    cdef int w_loc_j= w_loc_i
    cdef int w_loc_k= min(k_range,w_loc_i/2)
    
        
#    cdef np.ndarray[DTYPE_t1, ndim=2] Del_prop= np.array([[1,0], [0,1]], dtype= DTYPE1)
#    cdef np.ndarray[np.int_t, ndim=2, negative_indices=False, mode='c'] dk_randx = np.random.randint(m, a-m, (a,b))
#    cdef np.ndarray[np.int_t, ndim=2, negative_indices=False, mode='c'] dk_randy = np.random.randint(m, a-m, (a,b))
    

        
        
    # Local search
    
    for i in prange(m , a-m, nogil=True):
        for j in xrange(m, b-m):
            
            i_mn= max(m, i-w_loc_i)
            i_mx= min(a-m-1, i+w_loc_i)
            j_mn= max(m, j-w_loc_j)
            j_mx= min(b-m-1, j+w_loc_j)
            k_mn= max(0, k_ref-w_loc_k)
            k_mx= min(c-1, k_ref+w_loc_k)
            
            for i1 in xrange(i_mn, i_mx):
                for j1 in xrange(j_mn, j_mx):
                    for k1 in xrange(k_mn, k_mx+1):
                        
                        if NND[i,j,-1]>sim_threshold:
                        
                            new_rand= 1
                            if i1==i and j1==j and k1== k_ref:
                                new_rand= 0
                            
                            if new_rand==1:
                            
                                new_dist= 0
                                i11= i1-i
                                j11= j1-j
                                for ii in xrange(i-m,i+m+1):
                                    for jj in xrange(j-m,j+m+1):
                                        value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                        new_dist+= value1*value1
                                new_dist= new_dist**0.5
                                
                                if new_dist<sim_threshold:
                                    
                                    written= 0
                                    
                                    for iw in xrange(K):
                                        
                                        if written==0 and NND[i,j,iw]>sim_threshold:
                                            
                                            NNF[i,j,iw,0]= i1
                                            NNF[i,j,iw,1]= j1
                                            NNF[i,j,iw,2]= k1
                                            NND[i,j,iw]= new_dist
    
                                            written= 1
                            
                            
            
            

    


    # Propagation
            
            
    for i in prange(m, a-m-1, 2, nogil=True):
        for j in xrange(m, b-m-1):
            
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i1= NNF[i+1, j, k, 0]
                    j1= NNF[i+1, j, k, 1]
                    k1= NNF[i+1, j, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1


    for i in prange(m+1, a-m-1, 2, nogil=True):
        for j in xrange(m, b-m-1):
        
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i1= NNF[i+1, j, k, 0]
                    j1= NNF[i+1, j, k, 1]
                    k1= NNF[i+1, j, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1


    for j in prange(m, b-m-1, 2, nogil=True):
        for i in xrange(m, a-m-1):
                
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
        
                    i1= NNF[i, j+1, k, 0]
                    j1= NNF[i, j+1, k, 1]
                    k1= NNF[i, j+1, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1

        
    for j in prange(m+1, b-m-1, 2, nogil=True):
        for i in xrange(m, a-m-1):
                             
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
        
                    i1= NNF[i, j+1, k, 0]
                    j1= NNF[i, j+1, k, 1]
                    k1= NNF[i, j+1, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1


    for i in prange(m+1, a-m-1, 2, nogil=True):
        for j in xrange(m+1, b-m-1):
            
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i1= NNF[i-1, j, k, 0]
                    j1= NNF[i-1, j, k, 1]
                    k1= NNF[i-1, j, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1


    for i in prange(m+2, a-m-1, 2, nogil=True):
        for j in xrange(m+1, b-m-1):
            
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i1= NNF[i-1, j, k, 0]
                    j1= NNF[i-1, j, k, 1]
                    k1= NNF[i-1, j, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1


    for j in prange(m+1, b-m-1, 2, nogil=True):
        for i in xrange(m+1, a-m-1):
             
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
        
                    i1= NNF[i, j-1, k, 0]
                    j1= NNF[i, j-1, k, 1]
                    k1= NNF[i, j-1, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1

        
    for j in prange(m+2, b-m-1, 2, nogil=True):
        for i in xrange(m+1, a-m-1):
             
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
        
                    i1= NNF[i, j-1, k, 0]
                    j1= NNF[i, j-1, k, 1]
                    k1= NNF[i, j-1, k, 2]
                                                
                    new_rand= 1
                    if i1==i and j1==j and k1== k_ref:
                        new_rand= 0
                    else:
                        for kk in xrange(K):
                            if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                new_rand= 0
#                                break
                            
                    if new_rand==1:
                        
                        new_dist= 0
                        i11= i1-i
                        j11= j1-j
                        for ii in xrange(i-m,i+m+1):
                            for jj in xrange(j-m,j+m+1):
                                value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                new_dist+= value1*value1
                        new_dist= new_dist**0.5
                        
                        if new_dist<sim_threshold:
                            
                            written= 0
                            
                            for iw in xrange(K):
                                
                                if written==0 and NND[i,j,iw]>sim_threshold:
                                    
                                    NNF[i,j,iw,0]= i1
                                    NNF[i,j,iw,1]= j1
                                    NNF[i,j,iw,2]= k1
                                    NND[i,j,iw]= new_dist
    
                                    written= 1











   
#        # Random search
#            
#        for i in xrange(m,a-m-1):
#            for j in xrange(m,b-m-1):
#                
#                if search_mask[i,j]>0:
#                    
#                    for l in xrange(n_rand_search):
#                        
#    #                    i1= m + int(rand()/upper_x)
#    #                    j1= m + int(rand()/upper_y)
#                        i1= m + rand()% (a-2*m-1)
#                        j1= m + rand()% (b-2*m-1)
#                        k1= k_min + rand()% (k_max-k_min)
#
#                        
#                        new_rand= 1
#                        if i1==i and j1==j and k1== k_ref:
#                            new_rand= 0
#                        else:
#                            for kk in xrange(K):
#                                if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
#                                    new_rand= 0
#                                    break
#                        
#                        if new_rand==1:
#                        
#                            new_dist= 0
#                            i11= i1-i
#                            j11= j1-j
#                            for ii in xrange(i-m,i+m+1):
#                                for jj in xrange(j-m,j+m+1):
#                                    value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
#                                    new_dist+= value1*value1
#                            new_dist= new_dist**0.5
#                            
#                            new_indx=K
#                            while NND[i,j,new_indx-1]>new_dist and new_indx>0:
#                                new_indx+= -1
#                    
#                            if new_indx<K:
#                            
#                                for z in xrange(K-1,new_indx,-1):
#                                    
#                                    NNF[i,j,z,0]= NNF[i,j,z-1,0]
#                                    NNF[i,j,z,1]= NNF[i,j,z-1,1]
#                                    NNF[i,j,z,2]= NNF[i,j,z-1,2]
#                                    NND[i,j,z]= NND[i,j,z-1]
#            
#                                NNF[i,j,new_indx,0]= i1
#                                NNF[i,j,new_indx,1]= j1
#                                NNF[i,j,new_indx,2]= k1
#                                NND[i,j,new_indx]= new_dist









    # Forward enrichment and random search
    
    
#        if i_rad>8:
#            i_rad= i_rad/2
#            j_rad= j_rad/2
#            k_rad= k_rad/2
#            i_rad_2= i_rad/2
#            j_rad_2= j_rad/2
#            k_rad_2= k_rad/2
    
    









    for i in xrange(m,a-m):
        for j in xrange(m,b-m):
            
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i0= NNF[i,j,k,0]
                    j0= NNF[i,j,k,1]
                    k0= NNF[i,j,k,2]

                    if k0==k_ref:
                        
                        for l in xrange(K):
                            
                            i1= NNF[i0,j0,l,0]
                            j1= NNF[i0,j0,l,1]
                            k1= NNF[i0,j0,l,2]
        
                            new_rand= 1
                            if i1==i and j1==j  and k1== k_ref:
                                new_rand= 0
                            else:
                                for kk in xrange(K):
                                    if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                        new_rand= 0
#                                        break
                        
                            if new_rand==1:
                        
                                new_dist= 0
                                i11= i1-i
                                j11= j1-j
                                for ii in xrange(i-m,i+m+1):
                                    for jj in xrange(j-m,j+m+1):
                                        value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                        new_dist+= value1*value1
                                new_dist= new_dist**0.5
                                
                                if new_dist<sim_threshold:
                                    
                                    written= 0
                                    
                                    for iw in xrange(K):
                                        
                                        if written==0 and NND[i,j,iw]>sim_threshold:
                                            
                                            NNF[i,j,iw,0]= i1
                                            NNF[i,j,iw,1]= j1
                                            NNF[i,j,iw,2]= k1
                                            NND[i,j,iw]= new_dist
            
                                            written= 1
                    
                    

    for i in prange(m, a-m, nogil=True):
        for j in xrange(m,b-m):
            
            if NND[i,j,-1]>sim_threshold:
                
                for k in xrange(K):
                    
                    i0= NNF[i,j,k,0]
                    j0= NNF[i,j,k,1]
                    k0= NNF[i,j,k,2]
                    
                    for l in xrange(n_rand_search):
                        
                        
                        i1= i0- i_rad_2+ rand()% i_rad
                        if i1<m:
                            i1= m
                        if i1>a-m-1:
                            i1= a-m-1
                            
                        j1= j0- j_rad_2+ rand()% j_rad
                        if j1<m:
                            j1= m
                        if j1>b-m-1:
                            j1= b-m-1
                            
                        k1= k0- k_rad_2+ rand()% k_rad
                        if k1<k_min:
                            k1= k_min
                        if k1>k_max-1:
                            k1= k_max-1
                        
                        
                        new_rand= 1
                        if i1==i and j1==j and k1== k_ref:
                            new_rand= 0
                        else:
                            for kk in xrange(K):
                                if NNF[i,j,kk,0]==i1 and NNF[i,j,kk,1]==j1 and NNF[i,j,kk,2]==k1:
                                    new_rand= 0
#                                    break
                        
                        if new_rand==1:
                        
                            new_dist= 0
                            i11= i1-i
                            j11= j1-j
                            for ii in xrange(i-m,i+m+1):
                                for jj in xrange(j-m,j+m+1):
                                    value1= IMG[ii,jj,k_ref]- IMG2[ii+i11, jj+j11,k1]
                                    new_dist+= value1*value1
                            new_dist= new_dist**0.5
                            
                            if new_dist<sim_threshold:
                                
                                written= 0
                                
                                for iw in xrange(K):
                                    
                                    if written==0 and NND[i,j,iw]>sim_threshold:
                                        
                                        NNF[i,j,iw,0]= i1
                                        NNF[i,j,iw,1]= j1
                                        NNF[i,j,iw,2]= k1
                                        NND[i,j,iw]= new_dist
        
                                        written= 1
    
    
    
    
    
    
    
    
    for i in xrange(m,a-m):
        for j in xrange(m,b-m):
            
            if NND[i,j,-1]<sim_threshold:
                search_mask[i,j]= 0

#        if iter_count==5:
#            
#            dist_threshold= 0.0
#            
#            for i in xrange(m,a-m):
#                for j in xrange(m,b-m):
#                    for k in xrange(K):
#                        
#                        dist_threshold+= NND[i,j,k]
#    
#            dist_threshold= dist_threshold*dist_threshold_divisor
#                        
#            for i in xrange(m,a-m):
#                for j in xrange(m,b-m):
#                    
#                    dist_mean= 0.0
#                    
#                    for k in xrange(K):
#                        dist_mean+= NND[i,j,k]
#                    dist_mean= dist_mean/K
#                    
#                    if dist_mean<dist_threshold:
#                        search_mask[i,j]= 0
#    
#        elif iter_count>5:
#            
#            for i in xrange(m,a-m):
#                for j in xrange(m,b-m):
#                    
#                    if search_mask[i,j]>0:
#                        
#                        dist_mean= 0.0
#                        
#                        for k in xrange(K):
#                            dist_mean+= NND[i,j,k]
#                        dist_mean= dist_mean/K
#                        
#                        if dist_mean<dist_threshold:
#                            search_mask[i,j]= 0
    




    return NNF, NND, search_mask
