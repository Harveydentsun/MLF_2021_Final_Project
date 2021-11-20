# Quantitative Trading via Machine Learning in the Chinese Stock Market

## —— Course Final Project

### Machine Learning for Finance (FN 570) 2021-22 Module 1 (Fall 2021)

*author: Bo Sun, Xinran Guo*

* Jupyter notebook: [MLF Final Project](MLF_Final_Project.ipynb)
* Reference research paper: [Huatai paper](华泰人工智能系列三十二：alphanet：因子挖掘神经网络_2020-06-15_华泰证券.pdf)

------



### 1. Introduction

Using deep learning modle to extract features directly from raw data and make predictions in an end to end manner becomes more and more polular. In this project, we implement a kind of network structure in the research report mentioned above, directly extract the features in the quantity and price data, and the generated factors have a good performance. Basically, we stack the data together to form the data matrix which we call data picture in the following part, and apply two-dimensional Convolutional Neural Network (2d-CNN) to extract important feature from data picture. We train our model with historical stock data and conduct backtesting on rolling basis (see Figure 1). 

<img src="https://z3.ax1x.com/2021/06/22/RezGjA.png" style="" width="500">

### 2. Data collection

PHBS has a Oracle database for A share stock price, we get our data from here. Here is our data collection processing. 

- First we read from database.ini containing info of the database and use keys.
- Second we write queries to read data from Oracle database month by month (since there is a limit for query data size)
- Third we concatenate monthly data and save it as a parquet file.

*We save the data as parquet file and  everybody should be able to load the data from parquet file conveniently.* 

#### 2.1 Extra Collection

1. First is the ST stocks, where ST stands for "Special Treatment". The Shanghai and Shenzhen Stock Exchange conduct special treatment on the stock transactions of listed companies with abnormal financial conditions or other conditions, and prefix them with "ST" in front of the abbreviation, so such stocks are called ST stocks. We will exclude ST stocks from our portfolio because they are highly illiquid and facing great delisting risks, so we need to read them from database.
2. Next, we also need information on the listing and delisting date of stocks. China’s A-shares have a price limit system, and newly listed stocks will have consecutive up-limits for a period of time and cannot be traded. Based on the above reason, we also exclude stocks that have been listed for less than 120 days from our portfolio.

#### 2.2 Data Integration

1. merge all data into one single DataFrame
2. create a "isst" column indicating whether a stock is in Special Treatment
3. create a "istrade" column indicating whether a stock is traded on that day
4. create a "isnew" column indicating whether a stock has been traded for at least 120 days
5. drop additional columns ,convert date string to datetime type and fill nan columns
6. save the data as a parquet file

#### 2.3 Benchmark

We choose The Shanghai Securities Composite Index (000001.SH) as our benchmark. Its sample stocks are all stocks listed on the Shanghai Stock Exchange, including A shares and B shares, reflecting the changes in the prices of listed stocks on the Shanghai Stock Exchange. It was officially released on July 15, 1991.

*we also read the index data from PHBS database and save it as 'marketindex.parquet' file.*

### 3. Data Processing

In order to use CNN model, we need to convert our data to "data pictures" that serve as training and predicting inputs. Our data pictures take the following format:

**Stock universe :** All Chinese A-share main board stocks, excluding newly-listed stocks, ST and PT stocks, and all stocks that are suspended or hit price limits in the following trading day.

**Time span :** Starting from January 31, 2011 to May 29, 2020, we use past 1500 days data as input, among which 1200 days are training period, and the remaining 300 days are validation periods.

**Variables :**

- return: daily return
- open, high, low, close: daily open price, highest price, lowest price, and close price
- volume: daily trading volume
- vwap: daily volume weighted average price
- turn: daily turnover rate
- free_turn: daily turnover rate of free float shares

**Input shape :** 9*30

**Labels :** standardized returns in the next two weeks (10 days)

*Here is what our data pictures look like:*

<img src="https://z3.ax1x.com/2021/07/23/Wymq56.png" style="" width="300">

*Here is what our sample data pictures look like:*

```
Sample data picture: 
 [[ 1.48000000e+01  1.53000000e+01  1.52200000e+01  1.59000000e+01
   1.60400000e+01  1.62000000e+01  1.65600000e+01  1.73000000e+01
   1.74800000e+01  1.76400000e+01  1.79400000e+01  1.75300000e+01
   1.82000000e+01  1.80000000e+01  1.77600000e+01  1.75400000e+01
   1.77400000e+01  1.76500000e+01  1.77100000e+01  1.83500000e+01
   1.83700000e+01  1.77100000e+01  1.76700000e+01  1.80000000e+01
   1.82000000e+01  1.78100000e+01  1.74200000e+01  1.70800000e+01
   1.73800000e+01  1.77800000e+01]
 [ 1.52700000e+01  1.55500000e+01  1.60500000e+01  1.61100000e+01
   1.61200000e+01  1.69200000e+01  1.73700000e+01  1.81000000e+01
   1.76000000e+01  1.80000000e+01  1.85000000e+01  1.87800000e+01
   1.82900000e+01  1.80000000e+01  1.79000000e+01  1.79300000e+01
   1.83600000e+01  1.80500000e+01  1.83400000e+01  1.84800000e+01
   1.85000000e+01  1.77500000e+01  1.80000000e+01  1.85000000e+01
   1.83000000e+01  1.79400000e+01  1.74700000e+01  1.74300000e+01
   1.79300000e+01  1.85000000e+01]
 [ 1.48000000e+01  1.51300000e+01  1.52100000e+01  1.57700000e+01
   1.58000000e+01  1.61500000e+01  1.65400000e+01  1.73000000e+01
   1.72500000e+01  1.73300000e+01  1.73000000e+01  1.75300000e+01
   1.74500000e+01  1.75000000e+01  1.72900000e+01  1.73500000e+01
   1.76000000e+01  1.73300000e+01  1.77000000e+01  1.79600000e+01
   1.75400000e+01  1.72200000e+01  1.75400000e+01  1.79300000e+01
   1.76000000e+01  1.74500000e+01  1.66900000e+01  1.69000000e+01
   1.72500000e+01  1.77500000e+01]
 [ 1.51700000e+01  1.51800000e+01  1.59000000e+01  1.60600000e+01
   1.60300000e+01  1.65600000e+01  1.71000000e+01  1.74800000e+01
   1.75400000e+01  1.79100000e+01  1.75600000e+01  1.81300000e+01
   1.77000000e+01  1.77600000e+01  1.76300000e+01  1.77700000e+01
   1.77500000e+01  1.76300000e+01  1.79600000e+01  1.83200000e+01
   1.77000000e+01  1.76400000e+01  1.78400000e+01  1.81100000e+01
   1.78100000e+01  1.76600000e+01  1.71800000e+01  1.73700000e+01
   1.78300000e+01  1.84600000e+01]
 [ 1.51064000e+01  1.52927000e+01  1.57665000e+01  1.59986000e+01
   1.59674000e+01  1.65892000e+01  1.71273000e+01  1.77140000e+01
   1.74276000e+01  1.77094000e+01  1.76780000e+01  1.82845000e+01
   1.77425000e+01  1.77728000e+01  1.76278000e+01  1.76947000e+01
   1.79903000e+01  1.75821000e+01  1.80347000e+01  1.82411000e+01
   1.78987000e+01  1.75132000e+01  1.77503000e+01  1.81589000e+01
   1.78466000e+01  1.76784000e+01  1.69990000e+01  1.72163000e+01
   1.76979000e+01  1.82658000e+01]
 [ 1.21706482e+06  9.00425930e+05  1.59134715e+06  9.08819480e+05
   6.62562360e+05  1.60006232e+06  2.09561419e+06  2.01610552e+06
   9.60071950e+05  1.24456018e+06  1.89051905e+06  1.69850168e+06
   1.17559865e+06  1.03486504e+06  1.20582386e+06  8.46603620e+05
   1.00780383e+06  9.68452770e+05  9.57868630e+05  1.24763640e+06
   1.42946944e+06  8.48781530e+05  9.51424320e+05  1.02106281e+06
   9.40130070e+05  6.77258480e+05  1.28918923e+06  7.59856930e+05
   8.52930510e+05  1.37340072e+06]
 [ 2.50000000e+00  6.59000000e-02  4.74310000e+00  1.00630000e+00
  -1.86800000e-01  3.30630000e+00  3.26090000e+00  2.22220000e+00
   3.43200000e-01  2.10950000e+00 -1.95420000e+00  3.24600000e+00
  -2.37180000e+00  3.39000000e-01 -7.32000000e-01  7.94100000e-01
  -1.12500000e-01 -6.76100000e-01  1.87180000e+00  2.00450000e+00
  -3.38430000e+00 -3.39000000e-01  1.13380000e+00  1.51350000e+00
  -1.65650000e+00 -8.42200000e-01 -2.71800000e+00  1.10590000e+00
   2.64820000e+00  3.53340000e+00]
 [ 6.27200000e-01  4.64000000e-01  8.20000000e-01  4.68300000e-01
   3.41400000e-01  8.24500000e-01  1.07990000e+00  1.03890000e+00
   4.94700000e-01  6.41300000e-01  9.74200000e-01  8.75300000e-01
   6.05800000e-01  5.33300000e-01  6.21400000e-01  4.36300000e-01
   5.19300000e-01  4.99100000e-01  4.93600000e-01  6.42900000e-01
   7.36600000e-01  4.37400000e-01  4.90300000e-01  5.26200000e-01
   4.84500000e-01  3.49000000e-01  6.64300000e-01  3.91600000e-01
   4.39500000e-01  7.07700000e-01]
 [ 1.41500000e+00  1.04690000e+00  1.85020000e+00  1.05660000e+00
   7.70300000e-01  1.86030000e+00  2.43640000e+00  2.34400000e+00
   1.11620000e+00  1.44700000e+00  2.19800000e+00  1.97470000e+00
   1.36680000e+00  1.20320000e+00  1.40190000e+00  9.84300000e-01
   1.17170000e+00  1.12600000e+00  1.11370000e+00  1.45060000e+00
   1.66200000e+00  9.86800000e-01  1.10620000e+00  1.18710000e+00
   1.09300000e+00  7.87400000e-01  1.49890000e+00  8.83400000e-01
   9.91700000e-01  1.59680000e+00]]
```

### 4. Feature Engineering

#### 4.1 CNN

It is assumed that the past trading information indicates future return. The two-dimensional feature of CNN is desirable for this task because data pictures contain different kinds of variables and their time variation. However, CNN was originally designed to extract image information, which means CNN only focus on the partial  information of adjacent data points in our data pictures. So we've made some improvements here, we try different combinations of different rows of the data picture, and apply different functions to those combinations. Fianlly we get a collection of new data pictures derived from the different functions.  Here is the list of custom functions (Table 1).

$$\text{Table 1. List of Functions and Their Description}$$

| Function       | Description                                                  |  Mapping Example   |
| -------------- | :----------------------------------------------------------- | :----------------: |
| ts_corr        | compute the correlations between rows                        | (n,9,30)->(n,36,3) |
| ts_cov         | compute the covariance between rows                          | (n,9,30)->(n,36,3) |
| ts_mean        | compute the mean of blocks in each row                       | (n,9,30)->(n,9,3)  |
| ts_stddev      | compute the standard deviation of blocks in each row         | (n,9,30)->(n,9,3)  |
| ts_zscore      | compute the zscore (mean over std.dev.) of blocks in each row | (n,9,30)->(n,9,3)  |
| ts_return      | compute the return of blocks in each row                     | (n,9,30)->(n,9,3)  |
| ts_decaylinear | compute the weighted average (with decaying weights) of blocks in each row | (n,9,30)->(n,9,3)  |

The functions in Table 1 can be divided into two types. 

* The first type measures the relationship between rows. eg(ts_corr, ts_cov...)
*  The second type computes the statistics of each rows. eg(ts_mean, ts_stddev...)

Suppose we set stride (parameter in CNN) to be equal to 5.

1. The Figure 3 shows how the first type functions work. The function will slide between different rows which include adjacent rows and seperated rows to calculate the statistics like correlations and covariance (the number of rows of new data pictures becomes $C_{9}^{2}=36$). 
2. The Figure 4 shows how the first type functions work. The function will just slide across one row and calculate the statistics like mean and standard deviation (the number of rows of new data pictures is ​still 9).

<img src="https://z3.ax1x.com/2021/07/23/WsQEVA.png" style="" width="600">

<img src="https://z3.ax1x.com/2021/07/23/WsQNGV.png" style="" width="600">

#### 4.2 Batch Normalization

$$\text{Batch Mean : } \mu = \frac{1}{m} \sum_{l=0}^m Z^{l(i)}$$

$$\text{Batch Std. Dev. : } \sigma^2 = \frac{1}{m} \sum_{i=0}^m (Z^{l(i)}-\mu)^2$$

$$\text{Normalized Result : } \hat Z^{l(i)} = (Z^{l(i)} - \mu)/\sigma$$

where $Z^{l}$ denotes the layer and $Z^{l(i)}$ denotes the elements in the layer.

#### 4.3 Pipeline Structure

Here we combine CNN processing and Batch Normalization into Pipeline struction (Figure 5), we can see that each function generate a new data picture as our output. 

<img src="https://z3.ax1x.com/2021/07/23/Wrg3RK.png" style="" width="600">

### 5. Neural Network Architecture

#### 5.1 Pooling Layer

$$\text{Table 2. List of Pooling Functions and Their Description}$$

| Function | Description                            | Mapping Example |
| -------- | -------------------------------------- | --------------- |
| mean     | compute the mean of blocks in each row | (n,k,3)->(n,k)  |
| max      | compute the max of blocks in each row  | (n,k,3)->(n,k)  |
| min      | compute the min of blocks in each row  | (n,k,3)->(n,k)  |

#### 5.2 Model Structure

The network architecture is feature input layers + fully connected dense layer. The model structure of Stock-Selector-Alpha is shown below. After feature engineering layers, StockSelector-Alpha will flatten all features, disgarding the temporal information.

The flattened features are then fed into a dense layer with 64 units. The number of neurons in the hidden layer is not restrained, but the literature recommends setting it at an even power of 2, such as 64, 32, etc. for accelerating GPU/CPU calculations (Vanhoucke and Senior, 2011), and hence the 64 units.

<img src="https://github.com/XinranGuo/PHBS_MLF_2021/blob/main/Final_Project_Picture/model%20structure.png" width="">

#### 5.3 Model Setting

**Training Configuration**

* activation function for the dense layer is the linear rectified unit (ReLU), whose derivative is easy to compute.

* optimizer is a SGD algorithm called Adam, which is an extended version of SGD proposed by Kingma and Ba (2017).

* learning rate is 0.002

**Regularization**

Neural networks are extremely easy to overfit and regularization is indispensable.

We use three regularization methods from the literature to mitigate overfitting 

1. Early Stopping

   Early stopping is that we examine model performances on the validation set and stop training whenever we observe that model performances on the validation set stop improving. Typically, models will stop improving on the validation set earlier than on the training set, so it is named "Early Stopping".

2. Batch Normalization

   The rationale behind batch normalization is that Adam randomly selects a batch of observations and only uses that batch of data for gradient updating in every optimization round, because computing gradients for all observations is costly and unnecessary. Hence, sample draws may have heterogeneous distributions. In order to sterilize the impact of heterogeneous random draws, we normalize the random batch after the input layer and between each fully connected hidden layer. The specific operations are to first subtract the mean and then divide the square root of batch variance.

3. Dropout

   Dropout is a technique that literally “drop out” a certain portion of neurons. Dropout is a potent regularization method and has proved its value in computer vision. This can prevent the networks from getting too convoluted by randomly setting a specific proportion of neurons in each layer to 0. The dropout rate in the input layer is 20%, and in the hidden layer is 50%. It needs to be particularly emphasized that we enlarge the number of neurons accordingly. we adjust the number of neurons to be the original neuron number dividing the dropout rate. For example, if the dropout rate is 50% and the original number of neurons is 32, then the new number of neurons is 52/0.5 = 64.

### 6. Backtesting

#### 6.1 Baseline Model

In order to compare the performance of our model, we first constructs a baseline model using NN with 2 hiden layers.

**Structure**

- The first hidden layer consists of:
  - 90 neurons
  - ReLU activation function
  - 50% dropout
- The second hidden layer consists of:
  - 30 neurons
  - ReLU activation function
  - 50% dropout
- The output layer consists of:
  - 1 neurons
  - linear activation function

**Backtest Engine**

- **Initialize functions** : set parameters and load market data
  - we use VWAP as our trading price,
  - market price is adjusted for stock and cash dividends
- **Selecting function** : calculate target portfolio on a given trade day
  - we only select stocks that are tradable, not ST, not up-limit and not newly listed
- **Backtest function** : calculate portfolio netvalue and benchmark value
- **Evaluation function and plot function** : display the result
  - evaluation is displayed in a monthly basis
  - plot function plots the portfolio as well as benchmark

**Baseline Performace**

It can be observed from the following results that the baseline model perform just as good as our benchmark index 000001.SH.

|      | yearMonth |    return | annualReturn | maxDrawDown |     Sharpe |  winRate | excessReturn | annualExcessReturn |
| ---: | --------: | --------: | -----------: | ----------: | ---------: | -------: | -----------: | -----------------: |
|    0 |    202001 | -0.046604 |    -0.559254 |   -0.046604 | -15.315219 | 0.333333 |     -0.01487 |          -0.178444 |
|    1 |    202002 |  0.051131 |     0.613571 |   -0.053999 |   1.447644 |      0.7 |     0.104556 |            1.25467 |
|    2 |    202003 | -0.097726 |    -1.172713 |   -0.149902 |  -2.991332 |      0.5 |    -0.037124 |          -0.445485 |
|    3 |    202004 |  -0.00638 |    -0.076566 |   -0.067236 |  -0.379082 |  0.47619 |    -0.042573 |          -0.510874 |
|    4 |    202005 |  0.048727 |     0.584723 |   -0.044966 |     4.1436 | 0.611111 |     0.039719 |           0.476623 |
|    5 |    202006 |  0.116548 |     1.398571 |   -0.011351 |   10.43633 |     0.65 |     0.063216 |            0.75859 |
|    6 |    202007 |  0.153756 |     1.845073 |    -0.06958 |   5.656122 | 0.608696 |     0.059979 |           0.719745 |
|    7 |    202008 |  0.048246 |     0.578948 |   -0.058144 |   2.181778 | 0.714286 |     0.017945 |           0.215345 |
|    8 |    202009 | -0.067682 |    -0.812185 |   -0.092533 |  -4.763138 | 0.590909 |    -0.021304 |          -0.255646 |
|    9 |    202010 |  0.018418 |      0.22102 |    -0.06702 |   1.235858 |    0.625 |     0.003058 |           0.036697 |
|   10 |    202011 | -0.042985 |     -0.51582 |   -0.049109 |  -3.601519 | 0.333333 |    -0.076658 |          -0.919902 |
|   11 |    202012 | -0.025818 |    -0.309811 |   -0.045919 |  -3.020179 | 0.428571 |    -0.009915 |          -0.118982 |
|   12 |     Total |  0.126301 |     0.126301 |   -0.165017 |   0.533388 | 0.558442 |     0.027099 |           0.027099 |


<img src="https://github.com/XinranGuo/PHBS_MLF_2021/blob/main/Final_Project_Picture/baseline_model_picture.png" style="" width="">






