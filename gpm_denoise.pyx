# cython: profile=True

import numpy as np
cimport numpy as np
import cython
from cython.view cimport array as cvarray

DTYPE1 = np.int
ctypedef np.int_t DTYPE_t1
DTYPE2 = np.float
ctypedef np.float_t DTYPE_t2


@cython.boundscheck(False)
@cython.wraparound(False)
def gpm_denoise(float[:,:,::1] IMG, int[:,:,:,::1] NNF, float[:,:,::1] NND, float[:,::1] M_dct,float[:,::1] M_har_32,
                float[:,::1] M_har_16, float[:,::1] M_har_8, float[:,::1] M_har_4, float[:,::1] M_har_2, float[:,::1] M_kaiser,
                int m, int K, float ht_lambda, float exclusion_factor, int k_ref, int k_range):
    
    
    cdef int a = IMG.shape[0]
    cdef int b = IMG.shape[1]
    cdef int c = IMG.shape[2]
    cdef int n = 2*m+1
#    cdef float nnd_mean= NND[m:a-m,m:b-m].mean()
#    cdef float ht_lambda= 200.0
    
#    IMG_den_base = cvarray(shape=(a,b), itemsize=sizeof(float), format="f")
#    cdef float [:, ::1] IMG_den = IMG_den_base
#    ptch_ct_base = cvarray(shape=(a,b), itemsize=sizeof(float), format="f")
#    cdef float [:, ::1] ptch_ct = ptch_ct_base
#    Xn_base = cvarray(shape=(n,n,K), itemsize=sizeof(float), format="f")
#    cdef float [:, :, ::1] Xn = Xn_base
#    Xt_base = cvarray(shape=(n,n,K), itemsize=sizeof(float), format="f")
#    cdef float [:, :, ::1] Xt = Xt_base
#    Xd_base = cvarray(shape=(n,n,K), itemsize=sizeof(float), format="f")
#    cdef float [:, :, ::1] Xd = Xd_base
    


    cdef np.ndarray[DTYPE_t2, ndim=3] Xn= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xt= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xd= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xt1= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xd1= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xt2= np.zeros((n,n,K), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] Xd2= np.zeros((n,n,K), dtype= DTYPE2)

#    cdef np.ndarray[DTYPE_t2, ndim=1] d= np.zeros(K, dtype= DTYPE2)

    cdef int i, j, k, k_good, k1, k2, kk, n1, n2, ii, jj, ind, ind_x, ind_y, ind_p, w_ht, k_min, k_max
    cdef float value, value1, value2, value3, mean_up, mean_lo
    
    k_min= max(0, k_ref-k_range)
    k_max= min(c, k_ref+k_range+1)
    
    cdef np.ndarray[DTYPE_t2, ndim=3] IMG_den= np.zeros((a,b,k_max-k_min), dtype= DTYPE2)
    cdef np.ndarray[DTYPE_t2, ndim=3] ptch_ct= np.zeros((a,b,k_max-k_min), dtype= DTYPE2)
    
    
    
    for i in xrange(m,a-m):
        for j in xrange(m,b-m):
            
            
#            d=NND[i,j,:]
            

#            k_good= K
#            while k_good>2:
#                mean_up= 0
#                mean_lo= 0
#                for k in xrange(k_good/2):
#                    mean_lo+= NND[i,j,k]
#                    mean_up+= NND[i,j,k_good/2+k]
#                if mean_up>exclusion_factor*mean_lo:
#                    k_good/=2
#                else:
#                    break
            
            k_good= K
            while k_good>2:
                if NND[i,j,k_good-2]>exclusion_factor:
                    k_good/=2
                else:
                    break


            for ii in xrange(n):
                for jj in xrange(n):
                    Xn[ii,jj,0]= IMG[i-m+ii,j-m+jj,k_ref]
                    for k in xrange(k_good-1):
                        Xn[ii,jj,k+1]= IMG[NNF[i,j,k,0]-m+ii, NNF[i,j,k,1]-m+jj, NNF[i,j,k,2]]
                
#            Xn[:,:,0]= IMG[i-m:i+m+1, j-m:j+m+1]
#            for k in xrange(k_good-1):
#                Xn[:,:,k+1]= IMG[NNF[i,j,k,0]-m:NNF[i,j,k,0]+m+1, NNF[i,j,k,1]-m:NNF[i,j,k,1]+m+1]
            
            w_ht= 1
            
            if k_good==32:
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_x,ind]*Xn[ind,ind_y,ind_p]
                            Xt1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_y,ind]*Xt1[ind_x,ind,ind_p]
                            Xt2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_32[ind_p,ind]*Xt2[ind_x,ind_y,ind]
#                            Xt[ind_x,ind_y,ind_p]= value * ( (value**2) / (value**2+ht_lambda) )
#                            w_ht+= ( (value**2) / (value**2+ht_lambda) )**2
                            if value>ht_lambda or value<-ht_lambda:
                                Xt[ind_x,ind_y,ind_p]= value
                                w_ht+= 1
                            else:
                                Xt[ind_x,ind_y,ind_p]= 0
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_x]*Xt[ind,ind_y,ind_p]
                            Xd1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_y]*Xd1[ind_x,ind,ind_p]
                            Xd2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_32[ind,ind_p]*Xd2[ind_x,ind_y,ind]
                            Xd[ind_x,ind_y,ind_p]= value

            
            elif k_good==16:
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_x,ind]*Xn[ind,ind_y,ind_p]
                            Xt1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_y,ind]*Xt1[ind_x,ind,ind_p]
                            Xt2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_16[ind_p,ind]*Xt2[ind_x,ind_y,ind]
#                            Xt[ind_x,ind_y,ind_p]= value * ( (value**2) / (value**2+ht_lambda) )
#                            w_ht+= ( (value**2) / (value**2+ht_lambda) )**2
                            if value>ht_lambda or value<-ht_lambda:
                                Xt[ind_x,ind_y,ind_p]= value
                                w_ht+= 1
                            else:
                                Xt[ind_x,ind_y,ind_p]= 0
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_x]*Xt[ind,ind_y,ind_p]
                            Xd1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_y]*Xd1[ind_x,ind,ind_p]
                            Xd2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_16[ind,ind_p]*Xd2[ind_x,ind_y,ind]
                            Xd[ind_x,ind_y,ind_p]= value

            elif k_good==8:
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_x,ind]*Xn[ind,ind_y,ind_p]
                            Xt1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_y,ind]*Xt1[ind_x,ind,ind_p]
                            Xt2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_8[ind_p,ind]*Xt2[ind_x,ind_y,ind]
#                            Xt[ind_x,ind_y,ind_p]= value * ( (value**2) / (value**2+ht_lambda) )
#                            w_ht+= ( (value**2) / (value**2+ht_lambda) )**2
                            if value>ht_lambda or value<-ht_lambda:
                                Xt[ind_x,ind_y,ind_p]= value
                                w_ht+= 1
                            else:
                                Xt[ind_x,ind_y,ind_p]= 0
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_x]*Xt[ind,ind_y,ind_p]
                            Xd1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_y]*Xd1[ind_x,ind,ind_p]
                            Xd2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_8[ind,ind_p]*Xd2[ind_x,ind_y,ind]
                            Xd[ind_x,ind_y,ind_p]= value

            elif k_good==4:
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_x,ind]*Xn[ind,ind_y,ind_p]
                            Xt1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_y,ind]*Xt1[ind_x,ind,ind_p]
                            Xt2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_4[ind_p,ind]*Xt2[ind_x,ind_y,ind]
#                            Xt[ind_x,ind_y,ind_p]= value * ( (value**2) / (value**2+ht_lambda) )
#                            w_ht+= ( (value**2) / (value**2+ht_lambda) )**2
                            if value>ht_lambda or value<-ht_lambda:
                                Xt[ind_x,ind_y,ind_p]= value
                                w_ht+= 1
                            else:
                                Xt[ind_x,ind_y,ind_p]= 0
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_x]*Xt[ind,ind_y,ind_p]
                            Xd1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_y]*Xd1[ind_x,ind,ind_p]
                            Xd2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_4[ind,ind_p]*Xd2[ind_x,ind_y,ind]
                            Xd[ind_x,ind_y,ind_p]= value

            elif k_good==2:
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_x,ind]*Xn[ind,ind_y,ind_p]
                            Xt1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind_y,ind]*Xt1[ind_x,ind,ind_p]
                            Xt2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_2[ind_p,ind]*Xt2[ind_x,ind_y,ind]
#                            Xt[ind_x,ind_y,ind_p]= value * ( (value**2) / (value**2+ht_lambda) )
#                            w_ht+= ( (value**2) / (value**2+ht_lambda) )**2
                            if value>ht_lambda or value<-ht_lambda:
                                Xt[ind_x,ind_y,ind_p]= value
                                w_ht+= 1
                            else:
                                Xt[ind_x,ind_y,ind_p]= 0
                
                for ind_p in xrange(k_good):
                    for ind_y in xrange(n):
                        for ind_x in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_x]*Xt[ind,ind_y,ind_p]
                            Xd1[ind_x,ind_y,ind_p]= value
                
                for ind_p in xrange(k_good):
                    for ind_x in xrange(n):
                        for ind_y in xrange(n):
                            value= 0
                            for ind in xrange(n):
                                value+= M_dct[ind,ind_y]*Xd1[ind_x,ind,ind_p]
                            Xd2[ind_x,ind_y,ind_p]= value
                
                for ind_x in xrange(n):
                    for ind_y in xrange(n):
                        for ind_p in xrange(k_good):
                            value= 0
                            for ind in xrange(k_good):
                                value+= M_har_2[ind,ind_p]*Xd2[ind_x,ind_y,ind]
                            Xd[ind_x,ind_y,ind_p]= value


#            w_ht= w_ht**0.5
            
#            for ii in xrange(n):
#                for jj in xrange(n):
#                    IMG_den[i-m+ii,j-m+jj]+= Xd[ii,jj,0]*1.0/w_ht
#                    ptch_ct[i-m+ii,j-m+jj]+= 1.0/w_ht
#                    for k in xrange(k_good-1):
#                        if NNF[i,j,k,2]==k_ref:
#                            IMG_den[NNF[i,j,k,0]-m+ii, NNF[i,j,k,1]-m+jj]+= Xd[ii,jj,k+1]*1.0/w_ht
#                            ptch_ct[NNF[i,j,k,0]-m+ii, NNF[i,j,k,1]-m+jj]+= 1.0/w_ht



            for ii in xrange(n):
                for jj in xrange(n):
                    IMG_den[i-m+ii,j-m+jj, k_ref-k_min]+= Xd[ii,jj,0]*1.0/w_ht*M_kaiser[ii,jj]
                    ptch_ct[i-m+ii,j-m+jj, k_ref-k_min]+= 1.0/w_ht*M_kaiser[ii,jj]
                    for k in xrange(k_good-1):
#                        if NNF[i,j,k,2]==k_ref:
                        IMG_den[NNF[i,j,k,0]-m+ii, NNF[i,j,k,1]-m+jj, NNF[i,j,k,2]-k_min]+= Xd[ii,jj,k+1]*1.0/w_ht*M_kaiser[ii,jj]
                        ptch_ct[NNF[i,j,k,0]-m+ii, NNF[i,j,k,1]-m+jj, NNF[i,j,k,2]-k_min]+= 1.0/w_ht*M_kaiser[ii,jj]

#            IMG_den[i-m:i+m+1, j-m:j+m+1]+= Xd[:,:,0]*w_ht
#            ptch_ct[i-m:i+m+1, j-m:j+m+1]+= w_ht
#            
#            for k in xrange(k_good-1):
#                IMG_den[NNF[i,j,k,0]-m:NNF[i,j,k,0]+m+1, NNF[i,j,k,1]-m:NNF[i,j,k,1]+m+1]+= Xd[:,:,k+1]*w_ht
#                ptch_ct[NNF[i,j,k,0]-m:NNF[i,j,k,0]+m+1, NNF[i,j,k,1]-m:NNF[i,j,k,1]+m+1]+= w_ht
    

    for i in xrange(a):
        for j in xrange(b):
            for k in xrange(k_max-k_min):
                if ptch_ct[i,j,k]>0:
                    IMG_den[i,j,k]/= ptch_ct[i,j,k]
                else:
                    IMG_den[i,j,k]= IMG[i,j,k+k_min]
    
            
    return IMG_den, ptch_ct

    
    
      
    
    
    
    
    
    