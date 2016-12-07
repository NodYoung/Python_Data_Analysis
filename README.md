# Python_Data_Analysis
上了七月算法的python数据分析课程，把课上的一些代码资源整理了一下放在这里。

***课程内容***
**第1讲 Python入门**
安装Python与环境配置
Anaconda安装和使用
&emsp;&emsp;Jupyter Notebook
常用数据分析库Numpy、Scipy、Pandas和matplotlib安装和简介
&emsp;&emsp;Numpy
&emsp;&emsp;Scipy
&emsp;&emsp;Pandas
&emsp;&emsp;matplotlib
常用高级数据分析库nltk、igraph和scikit-learn介绍
&emsp;&emsp;ntlk
&emsp;&emsp;igraph
&emsp;&emsp;scikit-learn
Python2和Python3区别简介


**第2讲 准备数据与Numpy**
Numpy
&emsp;&emsp;简介
&emsp;&emsp;基本功能
&emsp;&emsp;效率对比
Numpy的ndarray 
创建ndarray
&emsp;&emsp;Numpy数据类型
&emsp;&emsp;数组与标量之间的运算
&emsp;&emsp;基本的索引和切片
&emsp;&emsp;布尔型索引
&emsp;&emsp;花式索引
&emsp;&emsp;数组转置和轴对称
&emsp;&emsp;快速的元素级数组函数
利用数组进行数据处理 
&emsp;&emsp;简介
&emsp;&emsp;将条件逻辑表述为数组运算
&emsp;&emsp;数学和统计方法
&emsp;&emsp;用于布尔型数组的方法
&emsp;&emsp;排序
&emsp;&emsp;去重以及其他集合运算
数组文件的输入输出
线性代数
随机数生成
高级应用 
&emsp;&emsp;数组重塑
&emsp;&emsp;数组的合并和拆分
&emsp;&emsp;元素的重复操作
&emsp;&emsp;花式索引的等价函数
例题分析 
&emsp;&emsp;距离矩阵计算


**第3讲 Python数据分析主力Pandas**
Pandas简介
基本功能
数据结构 
&emsp;&emsp;Series
&emsp;&emsp;DataFrame
&emsp;&emsp;索引对象
基本功能 
&emsp;&emsp;重新索引
&emsp;&emsp;丢弃指定轴上的项
&emsp;&emsp;索引、选取和过滤
&emsp;&emsp;算术运算和数据对齐
&emsp;&emsp;函数应用和映射
&emsp;&emsp;排序和排名
&emsp;&emsp;带有重复值的索引
汇总和计算描述统计
&emsp;&emsp;常用方法选项
&emsp;&emsp;常用描述和汇总统计函数
&emsp;&emsp;相关系数与协方差
&emsp;&emsp;唯一值以及成员资格
处理缺失数据
&emsp;&emsp;滤除缺失数据
&emsp;&emsp;填充缺失数据
层次化索引
&emsp;&emsp;重新分级顺序
&emsp;&emsp;根据级别汇总统计
&emsp;&emsp;使用DataFrame的列
其他话题
&emsp;&emsp;整数索引
&emsp;&emsp;面板（Pannel）数据


**第4讲 数据获取与处理**
多种格式数据加载、处理与存储
&emsp;&emsp;各式各样的文本数据
&emsp;&emsp;&emsp;&emsp;CSV与TXT读取
&emsp;&emsp;&emsp;&emsp;分片/块读取文本数据
&emsp;&emsp;&emsp;&emsp;把数据写入文本格式
&emsp;&emsp;&emsp;&emsp;手动读写数据（按要求）
&emsp;&emsp;&emsp;&emsp;JSON格式的数据
&emsp;&emsp;&emsp;&emsp;人人都爱爬虫，人人都要解析XML 和 HTML
&emsp;&emsp;&emsp;&emsp;解析XML
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;二进制格式的数据、使用HDF5格式、HTML与API交互
&emsp;&emsp;数据库相关操作
&emsp;&emsp;&emsp;&emsp;sqlite数据库
&emsp;&emsp;&emsp;&emsp;MySQL数据库
&emsp;&emsp;&emsp;&emsp;Memcache
&emsp;&emsp;&emsp;&emsp;MongoDB
Crawl and parsing HTML with Beauitful Soup
&emsp;&emsp;创建dataframe然后输出出来，为一会儿爬取做准备
&emsp;&emsp;Download the HTML and create a Beautiful Soup object
&emsp;&emsp;解析Beautiful Soup结构体
python正则表达式
&emsp;&emsp;学会用re.compile(strPattern[, flag])
&emsp;&emsp;Match
&emsp;&emsp;Pattern
&emsp;&emsp;match与search
&emsp;&emsp;split(string[, maxsplit]) | re.split(pattern, string[, maxsplit])
&emsp;&emsp;findall(string[, pos[, endpos]]) | re.findall(pattern, string[, flags])
&emsp;&emsp;finditer(string[, pos[, endpos]]) | re.finditer(pattern, string[, flags])
&emsp;&emsp;sub(repl, string[, count]) | re.sub(pattern, repl, string[, count])
&emsp;&emsp;subn(repl, string[, count]) |re.sub(pattern, repl, string[, count])
特征工程小案例：城市自行车共享系统使用状况

**第5讲 数据可视化Matplotlib**
Beyond柱状图：可视化能够为我们做些什么
&emsp;&emsp;可视化的理论介绍
&emsp;&emsp;&emsp;&emsp;圣经引用可视化；洞察数据内涵；寻找潜在模式
&emsp;&emsp;糟糕的可视化：一些具体案例
&emsp;&emsp;&emsp;&emsp;内容太多，KEEP IT SIMPLE STUPID
&emsp;&emsp;&emsp;&emsp;WRONG SCALE
&emsp;&emsp;&emsp;&emsp;乱用三维
&emsp;&emsp;&emsp;&emsp;少用3D，不要为了酷炫而酷炫
&emsp;&emsp;The Purpose of Data Visualization is to Convey Information to People
&emsp;&emsp;一些可视化设计原则
&emsp;&emsp;一些可视化场景
&emsp;&emsp;MORE THAN 2 DIMENSION
&emsp;&emsp;TREE MAP
可视化项目入门实战
&emsp;&emsp;如何使用python进行初步的可视化工作
&emsp;&emsp;&emsp;&emsp;学会从网上找资源
&emsp;&emsp;&emsp;&emsp;D3js.org ——> Visual Index
&emsp;&emsp;Coding实战
&emsp;&emsp;&emsp;&emsp;拿到数据，可视化看一看
知道画什么，比知道怎么画更重要！！！

**第6讲 使用NLTK进行Python文本分析**


**第7讲 Python社交网络分析igraph**
社交网络算法介绍
&emsp;&emsp;社交网络
&emsp;&emsp;社交网络算法应用场景
&emsp;&emsp;安装igraph
&emsp;&emsp;什么是图？
&emsp;&emsp;&emsp;&emsp;Undirected和Directed; Bipartite和Multigraph
&emsp;&emsp;图数据集
&emsp;&emsp;社交网络算法
&emsp;&emsp;&emsp;&emsp;分析指标
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;度
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;紧密中心性（closeness centrality）
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;介数中心性（betweenness centrality）
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;点介数
&emsp;&emsp;&emsp;&emsp;PageRank算法
&emsp;&emsp;&emsp;&emsp;社区发现算法
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;GN算法
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;GN算法-边介数（Betweenness）
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;GN算法-community_edge_betweenness
&emsp;&emsp;&emsp;&emsp;社区评价指标
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;模块度Modularity 
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Conductance
&emsp;&emsp;&emsp;&emsp;Louvain算法
&emsp;&emsp;&emsp;&emsp;LPA算法
&emsp;&emsp;&emsp;&emsp;SLPA算法
代码时间
&emsp;&emsp;Learn_igraph(net.data)
&emsp;&emsp;分析权利的游戏网络（stormofswords.csv）
社交网络算法在金融反欺诈中的应用
工具推荐


**第8讲 Python机器学习scikit-learn**
What is Machine Learning?
3 Types of Learning
Scikit-learn algorithm cheet-sheet
The simplest Sklearn workflow
Data Representation
Generation Synthetic Data
Supervised Workflow
Linear Regression
Unsupervised Transformers
Feature Scaling
Principal Component Analysis
K-means Clustering
Scikit-learn API
Preprocessing & Classification Overview
Holdout Evaluation
Holdout Validation
Learning Curves
Grid Search
Confusion Matrix
Support Vector Machines
&emsp;&emsp;Kernel Trick
Decision Trees
&emsp;&emsp;Classification & Continuous Features
&emsp;&emsp;Impurity measures
Deep learning
&emsp;&emsp;4 Key Factors that makes magic happens
&emsp;&emsp;Linear Models
&emsp;&emsp;Neural Networks
&emsp;&emsp;Inside a Neuron
&emsp;&emsp;Multi-layer NN
CNN 
&emsp;&emsp;key ideas
&emsp;&emsp;Dropout
&emsp;&emsp;Convolution layer
&emsp;&emsp;Case Study: 
&emsp;&emsp;&emsp;&emsp;LeNet-5, AlexNet, ZFNet, VGGNet, ResNet
&emsp;&emsp;Transfer Learning
&emsp;&emsp;Fool your Conv-net
RNN and Language Model
&emsp;&emsp;LSTM
&emsp;&emsp;Word2Vec


**第9讲 数据科学完整案例**
Word-cup-analysis
Ipython-soccer-predictions


**第10讲 Python分布式计算**
