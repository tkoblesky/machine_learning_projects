from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import random

'''

When you run the script it will generate three guassian distributed clusters as well as some outliers. It then attmepts to cluster the 
data and plot the results.

This code may be needed to run a few times as it is not finished. There should be three cluster centers and if the algorithm picks out more or less than 
three cluster centers. In each section there are what needs to be worked on to improve the code.
'''

plt.switch_backend('agg')

# inputs 
dc = 0.9 # changing dc changes the value of the density (p) array
dc2 = 0.01 # this threshhold is for excluding the 
p_thresh = 20.0 # changing the p and delta thresholds changes what is considered a cluster center
delta_thresh = 6.0

######################################################################################################################################################

# this is how the data is now created


#### things to change (import data as text files) ####

# create cluster 1
mean1 = [10, 5]
cov1 = [[1, 0], [0, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 200).T

# create cluster 2
mean2 = [-2, 4]
cov2 = [[1, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 200).T

# create cluster 3
mean3 = [1, -8]
cov3 = [[1, 0], [0, 1]]
x3, y3 = np.random.multivariate_normal(mean3, cov3, 200).T

# add outliers 
outliers = [[5,0],[12,-6],[-5,0]]
x_out = []; y_out = []
for idx in range(0,np.shape(outliers)[0]):
    x_out.append(outliers[idx][0])
    y_out.append(outliers[idx][1])
x_out = np.array(x_out)
y_out = np.array(y_out)

# stack the cluster arrays together
x_tot = np.hstack([x1,x2,x3,x_out])
y_tot = np.hstack([y1,y2,y3,y_out])

# shuffle the data around
combined = zip(x_tot, y_tot)
random.shuffle(combined)
x_tot[:], y_tot[:] = zip(*combined)

######################################################################################################################################################

# create the density (p) and the delta arrays

#### things to change (generalize to n-dimensional data, clean up the code, fix np.min(distances_temp) error by using try) ####

# create the p array
p = np.zeros(np.shape(y_tot)[0])
distances = []
for idx in range(0,np.shape(y_tot)[0]):
    x_temp = (x_tot[idx]-x_tot)**2
    y_temp = (y_tot[idx]-y_tot)**2
    # get rid of counting the same point
    x_temp = np.delete(x_temp,idx); y_temp = np.delete(y_temp,idx)
    distance = np.sqrt(x_temp+y_temp)
    distances.append(distance)
    # only count points that are closer than dc
    close = distance - dc
    close_sum = np.sum(close<0)
    p[idx] = close_sum
distances = np.array(distances)

# create the delta array
maxarg = np.argmax(p)
delta = np.zeros(np.shape(y_tot)[0])
for idx in range(0,np.shape(y_tot)[0]):
    if idx != maxarg:
        p_loop = p[idx]
        p_temp = np.delete(p,idx)
        distances_temp =  np.squeeze(distances[idx,np.where(p_temp>p_loop)])
        try:
            delta[idx] = np.min(distances_temp)
        except:
            print 'seems to be a max'
    else:
        delta[idx] = np.max(distances[idx,:])

######################################################################################################################################################

# find the cluster centers

#### things to change (generalize to n-dimension and arbitrary number of cluster centers, don't count outliers as a part of a cluster)
     
inds = np.squeeze(np.where((p>p_thresh) & (delta>delta_thresh)))

# if a cluster center is less than dc2, than exclude from list

distances2 = []
for idx in inds:
    x_temp2 = (x_tot[idx]-x_tot)**2
    y_temp2 = (y_tot[idx]-y_tot)**2
    distance2 = np.sqrt(x_temp2+y_temp2)
    distances2.append(distance2)
distances2 = np.array(distances2)
inds2 = np.argmin(distances2,axis=0)
print np.shape(inds)

######################################################################################################################################################

# this section is for plotting (only plots three cluster cneters so if that data will not show up)

fig1,axs1 = plt.subplots(1,3,figsize=(12,5))
axss1 = axs1.ravel()
fig1.subplots_adjust(left=0.08,bottom=0.11,right=0.98,top=0.88,wspace=0.31,hspace=0.20)

# plots for the three clusters individually (needs to be generalized to arbitrary number of clusters)
axss1[0].plot(x1, y1, 'x',c='c')
axss1[0].plot(x2, y2, 'o',c='r')
axss1[0].plot(x3, y3, 's',c='b')
axss1[0].plot(x_out, y_out, 'd',c='k')
axss1[0].axis('equal')

# plots the p vs delta graph
axss1[1].scatter(p,delta)

# plots the clustered data (needs to be generalized to arbitrary number of clusters)
axss1[2].plot(x_tot[inds2==0], y_tot[inds2==0], 'x',c='c')
axss1[2].plot(x_tot[inds2==1], y_tot[inds2==1], 'o',c='r')
axss1[2].plot(x_tot[inds2==2], y_tot[inds2==2], 's',c='b')
axss1[2].axis('equal')

# add the labels and such
axss1[0].set_title('All the data')
axss1[2].set_title('Clustered Data')
for idx in 0,2:
    axss1[idx].set_xlabel('x (arb units)')
    axss1[idx].set_ylabel('y (arb units)')
axss1[1].set_xlabel('p')
axss1[1].set_ylabel('$\delta$')

# plt.axis('equal')
#plt.show()
plt.savefig('temp.png')