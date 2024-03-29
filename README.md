# 项目说明
   本项目中的jupyter脚本并未记录代码的执行过程，仅供参考编写预处理原始数据、实现聚类算法的代码的方式。
## 目录
### 数据
原始数据下载地址：
https://pan.baidu.com/s/1uzYLjrFpbcIOdA5pobvLlg 提取密码：qmxk

整合后的数据(对原始数据进行整合、缺失值统计等分析)下载地址：
https://pan.baidu.com/s/179MYRqNzmgH6tI9g0LjtWw 提取密码：d93a

整合后的数据大部分是由原始数据中的**卡口状态**数据提取得到。同时，该文件夹也包括了卡口流量和卡口状态的对应信息文件。**注：卡口状态对应了各路段的旅行时间**

本实验中运用到的数据储存在“实验数据”文件夹中的train.npy,test.npy,train_stdn.npy,test_stdn.npy这些文件中。这些文件是基于挑选出的实验路段，并从整合后的数据中选取获得的。选取数据的特征在论文的“实验数据”的部分说明。在stdn实验部分，训练集为前31天，其中第1-3天不可用。测试集为总数据集的最后10天，其中第1-3天不可用。划分的依据可以参考STDN的原始论文的实验实施部分。

**生成实验数据的过程大致为：卡口状态数据-->data extract.npy-->整合后的数据-->EDA.npy-->Filter.npy-->get_inputs.npy->train.npy等实验数据**

### 脚本
1. 数据处理
    1. 清洗原始数据、调整数据的格式（间隔1分钟->间隔5分钟）：data extract.ipynb
    2. 对新数据集进行关于统计特征的分析：EDA（Exploratory Data Analysis）
    3. 筛选出拥堵路段（时速20km/h以下的比例超过50%）：Filter
    4. 得到模型的训练、测试集,并对两个数据集进行指数移动滑动平均法（EMA)：get_inputs
2. **核心部分：对路段进行聚类**
   1. 定义SFHC聚类算法和传统层次聚类的函数：utils/clustering.py
   2. 使用SFHC和其他聚类算法对数据集/交通网络进行聚类：hc.ipynb
   
  **聚类过程：train.npy-->mean_mat.npy-->hc.ipynb-->聚类结果** 
  
3. 搭建并训练结合聚类算法的各模型
   stdn.py,mlp.py,cnn.py,dpf.py 命令行的使用说明可以查看utils/basic_functions中get_arguments的定义。

        使用示例：
           * python stdn.py -m sfhc -t 70 -n stdn
           * python mlp.py -m hc -t 70 -n stdn
4. 测试模型
   mock.py 使用方法和训练模型的相同

    测试结果的可视化脚本：error_visualize.ipynb
### 可用模型
   用于测试的各模型放在models文件夹中。由于在训练STDN的模型时，无法保存模型的结构和权重。因此，在测试STDN模型时，选择先重新建立模型，再加载先前训练得到的权重。
   
   如果不需要使用训练好的模型，则无需下载models文件夹。
### 整合后的数据
    "实验数据"文件夹中包含了
    1. 从原始数据集中提取的训练集、测试集（STDN的训练集、测试集和其他模型有所不同，所以分开存放）。如test.npy、test_stdn.npy
    2. 原始数据集中各天对应的星期几：day_of_week.npy
### 备注
   STDN的原始论文中有其实验的GitHub链接，可以用于参考从而复现模型。（STDN的权重模型尚未从服务器中拉取下来）
   

