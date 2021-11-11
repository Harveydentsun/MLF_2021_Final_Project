#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm import tqdm

# In[2]:


def read_data():
    data = pd.read_parquet(r"data.parquet")
    factor_list = ['open','high','low','close','vwap','vol','pct','turnover','freeturn']
    data_picture = data[data.date >= pd.to_datetime('20190601')].set_index(['stockid','date'])
    data_picture = data_picture[factor_list]
    data_picture = data_picture.stack().unstack(level=1)
    data_pic_subsample=data_picture.iloc[:,:30]   # convert it into (9,30) for each picture
    data_pic_subsample=data_pic_subsample.dropna()  # drop nan values
    stock_id_list=data_pic_subsample.index.get_level_values(0).unique()

    sample=[]
    for ii in stock_id_list:
        sample.append(np.array(data_pic_subsample.loc[ii]))
    sample=np.array(sample)
    
    return sample


# In[3]:


#sample=read_data()


# In[4]:


#sample.shape


# In[5]:


#sample[0]


# ### feature extraction layer

# #### type 1: mapping is multi to one

# In[6]:


# ts_mean layer

def compute_mean(pic, window=10, stride=10):
    """
    compute the mean values from a single picture
    
    Parameters
    ----------
    pic : np.array of shape (9,30)

    Returns
    -------
    result : np.array of shape (9,3)
    
    """
    length=pic.shape[1]
    width=pic.shape[0]
    
    step=length//stride
    if length%stride!=0:
        step+=1
    
    result=np.zeros((width,step))
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[:,ii]=np.mean(pic[:,index_left:index_right],axis=1).copy()
        else:
            result[:,ii]=np.mean(pic[:,index_left:],axis=1).copy()
    return result


def ts_mean(sample,window=10, stride=10):
    """
    compute the mean pictures from the sample
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,9,3)
    
    Future improvement
    ------------------
    use a build-in function to replace the for-loop below
    
    """
    num=len(sample)
    
    result=[]
    for ii in range(num):
        pic=sample[ii,:,:].copy()
        result_temp=compute_mean(pic,window=window, stride=stride).copy()
        result.append(result_temp)
    result=np.array(result)
    
    return result


# In[7]:


# ts_stddev layer

def compute_stddev(pic, window=10, stride=10):
    """
    compute the std values from a single picture
    
    Parameters
    ----------
    pic : np.array of shape (9,30)

    Returns
    -------
    result : np.array of shape (9,3)
    
    """
    #automated version so that we can customize window and stride
    length=pic.shape[1]
    width=pic.shape[0]
    
    step=length//stride
    if length%stride!=0:
        step+=1
    
    result=np.zeros((width,step))
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[:,ii]=np.std(pic[:,index_left:index_right],axis=1).copy()
        else:
            result[:,ii]=np.std(pic[:,index_left:],axis=1).copy()
    return result


def ts_stddev(sample,window=10, stride=10):
    """
    compute the std pictures from the sample
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,9,3)
    
    Future improvement
    ------------------
    use a build-in function to replace the for-loop below
    
    """
    num=len(sample)
    
    result=[]
    for ii in range(num):
        pic=sample[ii,:,:].copy()
        result_temp=compute_stddev(pic,window=window, stride=stride).copy()
        result.append(result_temp)
    result=np.array(result)
    
    return result


# In[8]:


# ts_zscore

def compute_zscore(array, window=10, stride=10):
    """
    Parameters
    ----------
    array : 1-d np.array
    
    window: int
    
    stride: int

    Returns
    -------
    result : 1-d np.array
    
    """    
    length=len(array)
    
    step=length//stride
    if length%stride!=0:
        step+=1
    result=np.zeros(step)
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            array_temp=array[index_left:index_right].copy()
            result[ii]=array_temp.mean()/array_temp.std()
        else:
            array_temp=array[index_left:].copy()
            result[ii]=array_temp.mean()/array_temp.std()
    
    return result


def ts_zscore(sample,window=10, stride=10):
    """
    compute the zscore of the sample
    
    Parameters
    ----------
    sample : 3-d np.array of shape (n,9,30)
    window:int

    Returns
    -------
    result : 3-d np.array of shape (n,9,3)
    
    """ 
    result=np.apply_along_axis(compute_zscore, 2, sample, window, stride)
    return result


# In[9]:


# ts_decaylinear

def compute_decaylinear(array,window=10, stride=10, k=0.98):
    """
    compute the decaylinear of the input array
    
    decaylinear means weighted average of an array with more weights on recent data
    
    Parameters
    ----------
    array : 1-d np.array
    
    k: decay factor, should be between (0,1)
    
    window:int
    
    stride:int

    Returns
    -------
    result : 1-d np.array
    
    """ 
    length=len(array)
    step=length//stride
    if length%stride!=0:
        step+=1
    
    result=np.zeros(step)
    weight_raw=k**np.arange(window,0,-1)  # more weights on recent data
    weight=weight_raw/weight_raw.sum()
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[ii]=((array[index_left:index_right].copy())*weight).sum()
        else:
            num_temp=len(array[index_left:])  
            # we extract the last weights because the number of remaining values are less than window number
            weight_temp=weight_raw[-num_temp:]/weight_raw[-num_temp:].sum()
            result[ii]=((array[index_left:].copy())*weight_temp).sum()
    return result


def ts_decaylinear(sample,window=10, stride=10):
    """
    compute the decaylinear of the sample
    
    Parameters
    ----------
    sample : 3-d np.array of shape (n,9,30)
    
    window:int
    
    stride:int

    Returns
    -------
    result : 3-d np.array of shape (n,9,3)
    
    """ 
    result=np.apply_along_axis(compute_decaylinear, 2, sample, window, stride)
    return result


# In[10]:


#ts_return

def compute_return(array, window=10, stride=10):
    """
    Parameters
    ----------
    array : 1-d np.array
    
    window: int
    
    stride: int

    Returns
    -------
    result : 1-d np.array
    
    """    
    length=len(array)
    
    step=length//stride
    if length%stride!=0:
        step+=1
    result=np.zeros(step)
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            array_temp=array[index_left:index_right].copy()
            result[ii]=array_temp[-1]/array_temp[0]-1
        else:
            array_temp=array[index_left:].copy()
            result[ii]=array_temp[-1]/array_temp[0]-1
    
    return result


def ts_return(sample,window=10, stride=10):
    """
    compute the return of the sample
    
    Parameters
    ----------
    sample : 3-d np.array of shape (n,9,30)
    
    window: int
    
    stride: int

    Returns
    -------
    result : 3-d np.array of shape (n,9,3)
    
    """ 
    result=np.apply_along_axis(compute_return, 2, sample, window, stride)
    return result


# #### type 2: mapping is between rows

# In[11]:


# ts_corr layer

def extract_corr_array(mat):
    """
    compute the corr matrix from the matrix input
    then convert it into 1-d corr array in the same order as the research note
    
    Parameters
    ----------
    mat : np.array

    Returns
    -------
    result : 1-D np.array that describes the corr between rows
    
    """
    corr_mat=np.corrcoef(mat)
    index=np.triu_indices_from(corr_mat,1)  # obtain the index of upper triangular matric (excluding diagonal)
    return corr_mat[index]  # only return the array of the elements in upper triangular


def compute_corr(pic, window=10, stride=10):
    """
    compute the corr values from a single picture
    
    Parameters
    ----------
    pic : by default, np.array of shape (9,30)

    Returns
    -------
    result : np.array of shape (36,3)
    36 is because there is 36 relations between 9 rows
    
    """
    length=pic.shape[1]
    width=pic.shape[0]
    
    step=length//stride
    if length%stride!=0:
        step+=1
    
    result_width=int(width*(width-1)/2)
    result=np.zeros((result_width,step))
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[:,ii]=extract_corr_array(pic[:,index_left:index_right]).copy()
        else:
            result[:,ii]=extract_corr_array(pic[:,index_left:]).copy()
    return result


def ts_corr(sample,window=10, stride=10):
    """
    compute the corr pictures from the sample
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,36,3)
    
    Future improvement
    ------------------
    use a build-in function to replace the for-loop below
    
    """
    num=len(sample)
    
    result=[]
    for ii in range(num):
        pic=sample[ii,:,:].copy()
        result_temp=compute_corr(pic,window=window, stride=stride).copy()
        result.append(result_temp)
    result=np.array(result)
    return result


# In[12]:


# ts_cov layer

def extract_cov_array(mat):
    """
    compute the cov matrix from the matrix input
    then convert it into 1-d cov array in the same order as the research note
    
    Parameters
    ----------
    mat : np.array

    Returns
    -------
    result : 1-D np.array that describes the cov between rows
    
    """
    cov_mat=np.cov(mat)
    index=np.triu_indices_from(cov_mat,1)  # obtain the index of upper triangular matric (excluding diagonal)
    return cov_mat[index]  # only return the array of the elements in upper triangular


def compute_cov(pic, window=10, stride=10):
    """
    compute the cov values from a single picture
    
    Parameters
    ----------
    pic : by default, np.array of shape (9,30)

    Returns
    -------
    result : np.array of shape (36,3)
    36 is because there is 36 relations between 9 rows
    
    """
    length=pic.shape[1]
    width=pic.shape[0]
    
    step=length//stride
    if length%stride!=0:
        step+=1
    
    result_width=int(width*(width-1)/2)
    result=np.zeros((result_width,step))
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            try:  # there are bugs in sample[23]. rows have no variation
                result[:,ii]=extract_cov_array(pic[:,index_left:index_right]).copy()
            except:
                result[:,ii]=np.nan
        else:
            try:
                result[:,ii]=extract_cov_array(pic[:,index_left:]).copy()
            except:
                result[:,ii]=np.nan
    return result


def ts_cov(sample,window=10, stride=10):
    """
    compute the cov pictures from the sample
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,36,3)
    
    Future improvement
    ------------------
    use a build-in function to replace the for-loop below
    
    """
    num=len(sample)
    
    result=[]
    for ii in range(num):
        pic=sample[ii,:,:].copy()
        result_temp=compute_cov(pic,window=window, stride=stride).copy()
        result.append(result_temp)
    result=np.array(result)
    return result


# ### Batch normalization

# In[13]:


def BN(layer):
    """
    BN: Batch normalization
    
    normalize layer within batch
    
    Parameters
    ----------
    layer : np.array of shape (n,m,l)

    Returns
    -------
    BN_mat : np.array of shape (n,m,l)
    
    """    
    layer[np.isinf(layer)]=np.nan   # replace inf with nan
    mean_mat=np.nanmean(layer,axis=0)  # to exclude the impact of np.nan
    std_mat=np.nanstd(layer,axis=0)
    BN_mat=(layer-mean_mat)/std_mat
    return BN_mat


# ### pooling layer: mean, max, min

# In[14]:


def mean_process(array,window=3,stride=3):
    """
    compute the mean values from an array
    
    Parameters
    ----------
    array : one-dimensional np.array
    window : int, the length of piece for computing mean
    stride: int, how many units window moves at each time

    Returns
    -------
    result : np.array
    
    """
    length=len(array)
    step=length//stride
    if length%stride!=0:
        step+=1
    result=np.zeros(step)
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[ii]=np.mean(array[index_left:index_right]).copy()
        else:
            result[ii]=np.mean(array[index_left:]).copy()
    return result


def mean_pooling(layer,window=3,stride=3):
    """
    apply the mean process to each observation in the layer
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,k)
    
    Notice
    ------
    for sake of concatenation later, we flatten the feature for each observation
    
    """
    result=np.apply_along_axis(mean_process, 2, layer, window, stride)
    n=len(result)
    k=int(result.size/n)
    
    return result.reshape(n,k)


# In[15]:


def max_process(array,window=3,stride=3):
    """
    compute the max values from an array
    
    Parameters
    ----------
    array : one-dimensional np.array
    window : int, the length of piece for computing max
    stride: int, how many units window moves at each time

    Returns
    -------
    result : np.array
    
    """
    length=len(array)
    step=length//stride
    if length%stride!=0:
        step+=1
    result=np.zeros(step)
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[ii]=np.max(array[index_left:index_right]).copy()
        else:
            result[ii]=np.max(array[index_left:]).copy()
    return result


def max_pooling(layer,window=3,stride=3):
    """
    apply the max process to each observation in the layer
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,k)
    
    Notice
    ------
    for sake of concatenation later, we flatten the feature for each observation
    
    """
    result=np.apply_along_axis(max_process, 2, layer, window, stride)
    n=len(result)
    k=int(result.size/n)
    
    return result.reshape(n,k)


# In[16]:


def min_process(array,window=3,stride=3):
    """
    compute the min values from an array
    
    Parameters
    ----------
    array : one-dimensional np.array
    window : int, the length of piece for computing min
    stride: int, how many units window moves at each time

    Returns
    -------
    result : np.array
    
    """
    length=len(array)
    step=length//stride
    if length%stride!=0:
        step+=1
    result=np.zeros(step)
    
    for ii in range(step):
        index_left=ii*stride
        index_right=index_left+window
        if index_right<=length:
            result[ii]=np.min(array[index_left:index_right]).copy()
        else:
            result[ii]=np.min(array[index_left:]).copy()
    return result


def min_pooling(layer,window=3,stride=3):
    """
    apply the min process to each observation in the layer
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)

    Returns
    -------
    result : np.array of shape (n,k)
    
    Notice
    ------
    for sake of concatenation later, we flatten the feature for each observation
    
    """
    result=np.apply_along_axis(min_process, 2, layer, window, stride)
    n=len(result)
    k=int(result.size/n)
    
    return result.reshape(n,k)


# ### Main Program: Pipeline

# In[17]:


def pipeline(data,filter_list=[ts_corr,ts_cov,ts_stddev,ts_zscore,ts_return,ts_decaylinear,ts_mean],
             pooling_list=[mean_pooling,max_pooling,min_pooling]):
    
    """
    extract features from the input sample. The sample will go through each of the filters and pooling function.
    
    Batch Normalization is also applied after any function is applied to sample
    
    Parameters
    ----------
    sample : np.array of shape (n,9,30)
    
    filter_list: a list of filter functions that extract features from sample.
    
        By default, filter_list includes ts_corr,ts_cov,ts_stddev,ts_zscore,ts_decaylinear,ts_mean. 
    
        Each filter function is assumed with window=10 and stride=10.
    
    pooling_list:
    
        By default, pooling_list includes mean_pooling,max_pooling,min_pooling.

    Returns
    -------
    result : np.array of shape (n,k)
    
    """
    
    result_list=[]

    fil_num = 0
    for ii in tqdm(filter_list):
        print('filter number %d is starting'%fil_num)

        pooling_num = 0
        for jj in tqdm(pooling_list):
            print('pooling number %d is starting'%pooling_num)
            f_list=[ii,BN,jj,BN]
            final_result=data.copy()
            for f in f_list:
                final_result=f(final_result)
            result_list.append(final_result)
            pooling_num += 1

        print('filter number %d finished'%fil_num)
        print('\n')
        fil_num += 1

    result=np.concatenate(result_list,axis=1)
    return result




