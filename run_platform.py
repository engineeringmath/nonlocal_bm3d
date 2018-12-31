

import SimpleITK as sitk
import numpy as np
#import cProfile, pstats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

I= sitk.ReadImage("/home/davood/dk_data/Abdomen/RawData/Training/img/tmp-5054-1864591500.nii")
I= sitk.ReadImage("/home/davood/dk_data/Abdomen/RawData/Training/img/tmp-5054-1217466100.nii")

I= sitk.GetArrayFromImage(I)
I= np.transpose(I,[1,2,0])

import gpm_initialize
import gpm_improve
import gpm_denoise
import compute_nnd
import bm3d


m= 4
n= 2*m+1
K= 32


def display_knn(IMG,i,j,vij):
    
    fig,ax = plt.subplots(1)
    ax.imshow(IMG, cmap='gray')
    
    rect = patches.Rectangle((j-m-0.5,i-m-0.5),n,n,linewidth=1,edgecolor='r',facecolor='r')
    ax.add_patch(rect)
    
    for k in xrange(len(vij)):
        rect = patches.Rectangle((vij[k,1]-m-0.5,vij[k,0]-m-0.5),n,n,linewidth=1,edgecolor='b',facecolor='b')
        ax.add_patch(rect)
    
    plt.show()

def display_knn_multislice(IMG,i,j,vij, k_ref, k_range):
    
    for slice in xrange(k_ref-k_range, k_ref+k_range):
    
        fig,ax = plt.subplots(1)
        ax.imshow(IMG[:,:,slice], cmap='gray', vmin=-1000, vmax=1000)
       
        for k in xrange(len(vij)):
            if vij[k,2]==slice:
                rect = patches.Rectangle((vij[k,1]-m-0.5,vij[k,0]-m-0.5),n,n,linewidth=2,edgecolor='b',facecolor='none')
                ax.add_patch(rect)
                
        if slice==k_ref:
            rect = patches.Rectangle((j-m-0.5,i-m-0.5),n,n,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
        
        
        plt.show()



M_dct= np.zeros((n,n))
for k in xrange(n):
    for i in xrange(n):
        M_dct[k,i]= np.cos(np.pi/n*(i+1./2)*k)
M_dct[0,:]/=np.sqrt(2)
M_dct*= np.sqrt(2./n)

M_har_2= np.array([[1,1]/np.sqrt(2), 
                 [1,-1]/np.sqrt(2)])

M_har_4= np.vstack( ( np.kron(M_har_2,[1,1]) , np.kron(np.eye(2),[1,-1]) ) )
col_norms= np.linalg.norm(M_har_4, axis=1)
M_har_4= M_har_4 / col_norms[np.newaxis,:]

M_har_8= np.vstack( ( np.kron(M_har_4,[1,1]) , np.kron(np.eye(4),[1,-1]) ) )
col_norms= np.linalg.norm(M_har_8, axis=1)
M_har_8= M_har_8 / col_norms[np.newaxis,:]

M_har_16= np.vstack( ( np.kron(M_har_8,[1,1]) , np.kron(np.eye(8),[1,-1]) ) )
col_norms= np.linalg.norm(M_har_16, axis=1)
M_har_16= M_har_16 / col_norms[np.newaxis,:]

M_har_32= np.vstack( ( np.kron(M_har_16,[1,1]) , np.kron(np.eye(16),[1,-1]) ) )
col_norms= np.linalg.norm(M_har_32, axis=1)
M_har_32= M_har_32 / col_norms[np.newaxis,:]

v_kaiser= np.kaiser(n,2)
M_kaiser= np.outer(v_kaiser, v_kaiser)








IMG_true= I[60:390,:,:].copy().astype('float')

a,b,c= IMG_true.shape

IMG_noisy= IMG_true+ np.random.randn(a,b,c)*100

k_ref= 134
k_range= 1

nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref, k_range)

similarity_threshold= (2*81*10000)**0.5*1
max_iter= 20
search_mask= np.ones((a,b))
nnf_f, nnd_f, s_mask_up = gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                   max_iter, m, K, k_ref, k_range, similarity_threshold)

#statement = "nnf_f, nnd_f, s_mask_up= gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), max_iter, m, K, k_ref, k_range, similarity_threshold)"
#cProfile.runctx(statement, globals(), locals(), '.prof')
#s = pstats.Stats('.prof')
#s.strip_dirs().sort_stats('time').print_stats(30)

nnf_f= np.asarray(nnf_f).astype('int64')
nnd_f= np.asarray(nnd_f).astype('float64')
s_mask_up= np.asarray(s_mask_up).astype('float64')

#IMG_denoised= _gpm_denoise_cy.gpm_denoise(IMG_noisy, x11, x22, m, K)
IMG_den, p_count= gpm_denoise.gpm_denoise(IMG_noisy.astype('f'), nnf_f.astype('i'), nnd_f.astype('f'),
                                     M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                                        M_har_8.astype('f'), M_har_4.astype('f'), 
                                     M_har_2.astype('f'), M_kaiser.astype('f'), m, K, 250, 1100, k_ref, k_range)

np.linalg.norm(IMG_den[:,:,0]-IMG_true[:,:,k_ref])/(330*512)**0.5



import time

k_ref= 134
k_range= 5
n_iter= 20
nnd_mean= np.zeros(n_iter+1)
mis_mtch= np.zeros(n_iter+1)
msk_mean= np.zeros(n_iter+1)
lps_time= np.zeros(n_iter+1)


nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref, k_range)
similarity_threshold= (2*81*10000)**0.5
search_mask= np.ones((a,b))

t_i= time.time()

nnd_mean[0]= nnd_i[5:-5,5:-5,:].mean()
mis_mtch[0]= np.mean(nnd_i[5:-5,5:-5,:]>similarity_threshold)
msk_mean[0]= search_mask[5:-5,5:-5].mean()
lps_time[0]= time.time()- t_i

n_rand_search= 10
i_rad=32
w_loc_i=2

for i in xrange(n_iter):
    
    nnf_f, nnd_f, s_mask_up = gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                       10, m, K, k_ref, k_range, similarity_threshold, n_rand_search, i_rad, w_loc_i)
    nnf_f= np.asarray(nnf_f).astype('int64')
    nnd_f= np.asarray(nnd_f).astype('float64')
    s_mask_up= np.asarray(s_mask_up).astype('float64')
    
    nnd_mean[i+1]= nnd_f[5:-5,5:-5,:].mean()
    mis_mtch[i+1]= np.mean(nnd_f[5:-5,5:-5,:]>similarity_threshold)
    msk_mean[i+1]= s_mask_up[5:-5,5:-5].mean()
    lps_time[i+1]= time.time()- t_i
    
    nnf_i, nnd_i, search_mask= nnf_f, nnd_f, s_mask_up
    
    print i, s_mask_up[5:-5,5:-5].mean(), time.time()- t_i, nnd_mean[i+1]
    







m= 4
n= 2*m+1
K= 16


k_ref= 52
k_range= 5
n_iter= 10


n_rand_search_v= [2, 8, 32]
i_rad_v= [2, 8, 32]
w_loc_i_v= [2, 4, 16]


NND_MEAN= np.zeros((n_iter+1, len(n_rand_search_v), len(i_rad_v), len(w_loc_i_v)))
MIS_MTCH= np.zeros((n_iter+1, len(n_rand_search_v), len(i_rad_v), len(w_loc_i_v)))
MSK_MEAN= np.zeros((n_iter+1, len(n_rand_search_v), len(i_rad_v), len(w_loc_i_v)))
LPS_TIME= np.zeros((n_iter+1, len(n_rand_search_v), len(i_rad_v), len(w_loc_i_v)))

###
nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref-1, k_range)
nnf_f, nnd_f, s_mask_up = gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                       40, m, K, k_ref, k_range, similarity_threshold, n_rand_search, i_rad, w_loc_i)
nnf_f= np.asarray(nnf_f).astype('int64')
nnd_f= np.asarray(nnd_f).astype('float64')
nnf_ii= nnf_f
nnf_ii[:,:,:,2]+= 1
_ , nnd_ii= compute_nnd.compute_nnd(IMG_noisy, nnf_ii, m, K, k_ref)
##

for i1 in xrange(len(n_rand_search_v)):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            
            n_rand_search= n_rand_search_v[i1]
            i_rad= i_rad_v[i2]
            w_loc_i= w_loc_i_v[i3]
            
            nnd_mean= np.zeros(n_iter+1)
            mis_mtch= np.zeros(n_iter+1)
            msk_mean= np.zeros(n_iter+1)
            lps_time= np.zeros(n_iter+1)
            
#            nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref, k_range)
            nnd_i= nnd_ii.copy()
            nnf_i= nnf_ii.copy()
            similarity_threshold= (2*81*10000)**0.5
            search_mask= np.ones((a,b))
            
            t_i= time.time()
            
            nnd_mean[0]= nnd_i[5:-5,5:-5,:].mean()
            mis_mtch[0]= np.mean(nnd_i[5:-5,5:-5,:]>similarity_threshold)
            msk_mean[0]= search_mask[5:-5,5:-5].mean()
            lps_time[0]= time.time()- t_i
            
            for i in xrange(n_iter):
                
                if i>0:
                    w_loc_i=0
                
                nnf_f, nnd_f, s_mask_up = gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                                   1, m, K, k_ref, k_range, similarity_threshold, n_rand_search, i_rad, w_loc_i)
                nnf_f= np.asarray(nnf_f).astype('int64')
                nnd_f= np.asarray(nnd_f).astype('float64')
                s_mask_up= np.asarray(s_mask_up).astype('float64')
                
                nnd_mean[i+1]= nnd_f[5:-5,5:-5,:].mean()
                mis_mtch[i+1]= np.mean(nnd_f[5:-5,5:-5,:]>similarity_threshold)
                msk_mean[i+1]= s_mask_up[5:-5,5:-5].mean()
                lps_time[i+1]= time.time()- t_i
                
                nnf_i, nnd_i, search_mask= nnf_f, nnd_f, s_mask_up
            
            NND_MEAN[:,i1,i2,i3]= nnd_mean
            MIS_MTCH[:,i1,i2,i3]= mis_mtch
            MSK_MEAN[:,i1,i2,i3]= msk_mean
            LPS_TIME[:,i1,i2,i3]= lps_time
            
            print i1, i2, i3
    



for i1 in xrange(len(n_rand_search_v)):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            
            print '{0: <15}'.format(n_rand_search_v[i1]),  \
                  '{0: <15}'.format(i_rad_v[i2]),  \
                  '{0: <15}'.format(w_loc_i_v[i3]), \
                  '{0: <15}'.format(NND_MEAN[1,i1,i2,i3]), \
                  '{0: <15}'.format(MIS_MTCH[1,i1,i2,i3]), \
                  '{0: <15}'.format(MSK_MEAN[1,i1,i2,i3]), \
                  '{0: <15}'.format(LPS_TIME[1,i1,i2,i3])


x= LPS_TIME
y= MIS_MTCH
plt.figure()
for i1 in xrange(2,3):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            plt.plot(x[:,i1,i2,i3], y[:,i1,i2,i3], '.-', \
                     label="n_rand "+ str(n_rand_search_v[i1])\
                            +", i_rad "+ str(i_rad_v[i2]) \
                            +", w_loc "+ str(w_loc_i_v[i3]) )
plt.legend()







x= LPS_TIME
y= MIS_MTCH
Y= np.zeros( (len(n_rand_search_v) * len(i_rad_v) * len(w_loc_i_v)*6, 11) )
X= np.zeros( (11, len(n_rand_search_v) * len(i_rad_v) * len(w_loc_i_v)*3) )
ind= -3

for i1 in xrange(len(n_rand_search_v)):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            ind+= 3
            X[:,ind]= x[:,i1,i2,i3]
            X[:,ind+1]= y[:,i1,i2,i3]

            











y= MSK_MEAN
plt.figure()
for i1 in xrange(0,1):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            x= np.linspace(0,n_iter, n_iter+1)
            plt.plot(x, y[:,i1,i2,i3], '.-', \
                     label="n_rand "+ str(n_rand_search_v[i1])\
                            +", i_rad "+ str(i_rad_v[i2]) \
                            +", w_loc "+ str(w_loc_i_v[i3]) )
plt.legend()







y= LPS_TIME
plt.figure()
for i1 in xrange(2,3):
    for i2 in xrange(len(i_rad_v)):
        for i3 in xrange(len(w_loc_i_v)):
            x= np.linspace(1,n_iter, n_iter)
            plt.plot(x, np.diff(y[:,i1,i2,i3]), '.-', \
                     label="n_rand "+ str(n_rand_search_v[i1])\
                            +", i_rad "+ str(i_rad_v[i2]) \
                            +", w_loc "+ str(w_loc_i_v[i3]) )
plt.legend()









k_ref= 133
k_range= 0
n_iter= 10


n_rand_search_v= [1, 3, 5, 10, 20]
i_rad_v=[4, 8, 16, 32]
w_loc_i_v= [4, 8, 16]

i1= 2
i2= 1
i3= 1

n_rand_search= n_rand_search_v[i1]
i_rad= i_rad_v[i2]
w_loc_i= w_loc_i_v[i3]

similarity_threshold= (2*81*10000)**0.5
search_mask= np.ones((a,b))


import gpm_improve_par

nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref, k_range)
nnf_f, nnd_f, s_mask_up = gpm_improve_par.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                                   1, m, K, k_ref, k_range, similarity_threshold, n_rand_search, i_rad, w_loc_i)









#####   boosting etc.




IMG_buff= np.zeros(IMG_noisy.shape)
WGH_buff= np.zeros(IMG_noisy.shape)

k_range= 0
s_start= 130
s_end= 140
s_step= 1

nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, s_start, k_range)

search_mask= np.ones((a,b))
max_iter= 50
exclusion_factor= (2*81*10000)**0.5*1.05
nnf_f, nnd_f, s_mask_up= gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                   max_iter, m, K, s_start, k_range, exclusion_factor)

nnf_f= np.asarray(nnf_f).astype('int64')
nnd_f= np.asarray(nnd_f).astype('float64')
s_mask_up= np.asarray(s_mask_up).astype('float64')

IMG_den, p_count= gpm_denoise.gpm_denoise(IMG_noisy.astype('f'), nnf_f.astype('i'), nnd_f.astype('f'),
                                     M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                                        M_har_8.astype('f'), M_har_4.astype('f'), 
                                     M_har_2.astype('f'), M_kaiser.astype('f'), m, K, 275, exclusion_factor, s_start, k_range)

IMG_buff[:,:,s_start-k_range:s_start+k_range+1] += IMG_den*p_count
WGH_buff[:,:,s_start-k_range:s_start+k_range+1] += p_count


for s_cur in xrange(s_start+s_step, s_end, s_step):
    
    nnf_i= nnf_f
    
    nnf_i[:,:,:,2]+= s_step
    _ , nnd_i= compute_nnd.compute_nnd(IMG_noisy, nnf_i, m, K, s_cur)
    
    max_iter= 20
    search_mask= np.ones((a,b))
    nnf_f, nnd_f, s_mask_up= gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                   max_iter, m, K, s_cur, k_range, exclusion_factor)
    
    nnf_f= np.asarray(nnf_f).astype('int64')
    nnd_f= np.asarray(nnd_f).astype('float64')
    
#    IMG_noisy_temp= IMG_noisy+IMG_buff
    
    IMG_den, p_count= gpm_denoise.gpm_denoise(IMG_noisy.astype('f'), nnf_f.astype('i'), nnd_f.astype('f'),
                                     M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                                        M_har_8.astype('f'), M_har_4.astype('f'), 
                                     M_har_2.astype('f'), M_kaiser.astype('f'), m, K, 250, 1100, s_cur, k_range)
    
    IMG_buff[:,:,s_cur-k_range:s_cur+k_range+1] += IMG_den*p_count
    WGH_buff[:,:,s_cur-k_range:s_cur+k_range+1] += p_count

    print s_cur


for s_cur in xrange(s_start+s_step, s_end, s_step):
    for i in xrange(a):
        for j in xrange(b):
            if WGH_buff[i,j,s_cur]>0:
                IMG_buff[i,j,s_cur]/= WGH_buff[i,j,s_cur]
            else:
                IMG_buff[i,j,s_cur]= IMG_noisy[i,j,s_cur]








HT= np.array([250])
EX= np.array([700, 1500])

Err=     np.zeros((len(HT),len(EX)))
IM_DEN=  np.zeros((a,b, len(HT),len(EX)))
for i in xrange(len(HT)):
    for j in xrange(len(EX)):
        IMG_den, p_count= gpm_denoise.gpm_denoise(IMG_noisy.astype('f'), nnf_f.astype('i'), nnd_f.astype('f'),
                                     M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                                        M_har_8.astype('f'), M_har_4.astype('f'), 
                                     M_har_2.astype('f'), M_kaiser.astype('f'), m, K, HT[i], EX[j], s_start, k_range)
        Err[i,j]= np.linalg.norm(IMG_true[10:-10,10:-10,k_ref]- IMG_den[10:-10,10:-10,5])/(310*492)**0.5
        IM_DEN[:,:,i,j]= IMG_den[:,:,5]
        print i,j



        
        
fig = plt.figure()
ax = fig.add_subplot(111)


#plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')
plt.imshow(np.abs(IM_DEN[:,:,2,5]/IM_DEN[:,:,2,0]-1), vmin= 0.0, vmax= .01)


fig, ax = plt.subplots()
cax = ax.imshow(np.abs(IM_DEN[:,:,0,1]/IM_DEN[:,:,0,0]-1), vmin= 0.0, vmax= .05321)
#cbar = fig.colorbar(cax, ticks=[5, 15, 30])
#cbar.ax.set_yticklabels(['5', '15', '30'])  # vertically oriented colorbar
plt.show()


for i in xrange(7):
    fig, ax = plt.subplots()
    cax = ax.imshow(np.abs(IM_DEN[:,:,4,8]/IM_DEN[:,:,4,i]-1), vmin= 0.0, vmax= .01)
    cbar = fig.colorbar(cax, ticks=[5, 15, 30])
    cbar.ax.set_yticklabels(['5', '15', '30'])  # vertically oriented colorbar
    plt.show()











# BM3D

import time

Nstep= 3
tau_match= 2500
lambda3d_ht= 2.7
sigma_est= 100
Ns= 59
nnf_bm3d= 100*np.ones((a,b,K,2))
nnd_bm3d= 1e10*np.ones((a,b,K))
k_ref= 135

t_beg= time.time()

IMG_bm3d_1= bm3d.bm3d_denoise_ht(IMG_noisy[:,:,k_ref].astype('f'), nnf_bm3d.astype('i'), nnd_bm3d.astype('f'), 
                            M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                            M_har_8.astype('f'), M_har_4.astype('f'), M_har_2.astype('f'), M_kaiser.astype('f'),
                            m, K, Nstep, Ns, lambda3d_ht, tau_match, sigma_est)

print time.time() - t_beg

np.linalg.norm(IMG_bm3d_1-IMG_true[:,:,k_ref])/(a*b)**0.5


tau_match= 400
K= 32
nnf_bm3d= 100*np.ones((a,b,K,2))
nnd_bm3d= 1e10*np.ones((a,b,K))

IMG_bm3d_2= bm3d.bm3d_denoise_wn(IMG_noisy[:,:,k_ref].astype('f'), IMG_bm3d_1.astype('f'), nnf_bm3d.astype('i'), 
                            nnd_bm3d.astype('f'), 
                            M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                            M_har_8.astype('f'), M_har_4.astype('f'), M_har_2.astype('f'), M_kaiser.astype('f'),
                            m, K, Nstep, Ns, tau_match, sigma_est)




#for i in xrange(20):
#    IMG_den= gpm_denoise_new.gpm_denoise(IMG_noisy.astype('f'), x11.astype('i'), x22.astype('f'),
#                                     M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), M_har_8.astype('f'), M_har_4.astype('f'), 
#                                     M_har_2.astype('f'),  m, K, 200+i*10, 1.2)
#    print 200+i*10, np.linalg.norm(IMG_den[:,:]-IMG_true[:,:])/199


#x1, x2= _gpm_initialize_cy.gpm_init(IMG_noisy, m, K)
#max_it= 100
#err= np.zeros(max_it)
#for i in xrange(max_it):
#    x10= x1.copy()
#    x20= x2.copy()
##    x1, x2= gpm_improve_new.gpm_iter(IMG_noisy, x1, x2, 1, m, K)
#    x1, x2= gpm_improve_new.gpm_iter(IMG_noisy.astype('f'), x1.astype('i'), x2.astype('f'), 1, m, K)
#    x1= np.asarray(x1).astype('int64')
#    x2= np.asarray(x2).astype('float64')
#    err[i]= x2[m:a-m,m:b-m].mean()
#    if i%10==100:
#        x2plot= np.ones((a,b))
#        x2plot[np.mean(x2,axis=2)>1272.8]=0
#        plt.figure(), plt.imshow(x2plot, cmap='gray')
#        print x2[m:a-m,m:b-m].mean(), np.linalg.norm(x1-x10), np.sum(x20-x2)

 
        
#
#new_indx= np.argmax(NND[i,j,:]>new_dist)
#
#new_indx=K
#while NND[i,j,new_indx-1]>new_dist and new_indx>0:
#    new_indx+= -1
#    















IMG_true= I[60:390,:,:].copy().astype('float')

a,b,c= IMG_true.shape

IMG_noisy= IMG_true+ np.random.randn(a,b,c)*100

k_ref= 135
k_range= 0
K= 32

factor= np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4 ])

IMAGE= np.zeros((a,b,len(factor)))
NND_min= np.zeros((a,b,len(factor)))
NND_max= np.zeros((a,b,len(factor)))
NND_mean= np.zeros((a,b,len(factor)))
MASK= np.zeros((a,b,len(factor)))


for ii in xrange(len(factor)):
    
    nnf_i, nnd_i= gpm_initialize.gpm_init(IMG_noisy, m, K, k_ref, k_range)
    
    similarity_threshold= (2*81*10000)**0.5*factor[ii]
    max_iter= 1
    search_mask= np.ones((a,b))
    nnf_f, nnd_f, s_mask_up = gpm_improve.gpm_iter(IMG_noisy.astype('f'), IMG_noisy.astype('f'), nnf_i.astype('i'), nnd_i.astype('f'), search_mask.astype('i'), 
                                       max_iter, m, K, k_ref, k_range, similarity_threshold)
    
    nnf_f= np.asarray(nnf_f).astype('int64')
    nnd_f= np.asarray(nnd_f).astype('float64')
    s_mask_up= np.asarray(s_mask_up).astype('float64')
    
    IMG_den, p_count= gpm_denoise.gpm_denoise(IMG_noisy.astype('f'), nnf_f.astype('i'), nnd_f.astype('f'),
                                         M_dct.astype('f'), M_har_32.astype('f'), M_har_16.astype('f'), 
                                            M_har_8.astype('f'), M_har_4.astype('f'), 
                                         M_har_2.astype('f'), M_kaiser.astype('f'), m, K, 250, 1273, k_ref, k_range)
    
    print  np.linalg.norm(IMG_den[10:-10,10:-10,0]-IMG_true[10:-10,10:-10,k_ref])/(310*492)**0.5

    IMAGE[:,:,ii]= IMG_den[:,:,0]
    NND_min[:,:,ii]= np.min(nnd_f, axis= 2)
    NND_max[:,:,ii]= np.max(nnd_f, axis= 2)
    NND_mean[:,:,ii]= np.mean(nnd_f, axis= 2)
    MASK[:,:,ii]= s_mask_up












