### 1. 安装环境
在当前目录下执行如下命令安装项目依赖环境：
```shell
$ conda create -n tianchi python=3.7
$ conda activate tianchi
$ pip install -r requirements.txt
```
### 2. 数据预处理
#数据下载地址
数据链接: https://pan.baidu.com/s/1pXq9VJAscubOC4Wxk2Effg  密码: 17bt
数据下载之后，根据数据名字中的round1和round2分别放到tcdata/nlp_round1_data和tcdata/nlp_round2_data下

进入 code 目录，运行下面的命令
```shell
$ bash preprocess.sh
```
主要生成bert训练数据和vocab，并使用数据对偶的方法最终得到 250000 条 训练数据，和加入特殊符号的vocab词典。  

根据给出的代码规范中命名的方式，将 b 榜测试数据放在 tcdata/nlp_round1_data 下。

### 3. 训练
选用了三种模型进行与训练，第一种是是采用全词mask的bert模型，选用的预训练模型是[bert-base-chinese],
第二种是使用基础的nezha模型，选用的预训练模型是[nezha-ch-base]
第三种是使用ngram(ngram mask策略)nezha模型,选用的预训练模型是[nezha-ch-base]

1. 进入 code 目录，执行如下命令进行第一阶段训练：
    ```shell
    $ bash pretrain_stage.sh
    ```
    这个阶段共训练 3个模型，在每个预训练模型中每次与当前最佳得分进行比较，保证保存的模型最佳，
    最终得到的3个预训练模型
    这一阶段已经完成，代码中已经提供了训练好的模型
    
2. 用经过第一阶段训练的最好模型初始化第二阶段的模型，即将 finetune_stage.sh 中的 
model_path 和 tokenizer_dir 换成真实的第一阶段最佳模型目录，可以获得三个模型对于b榜数据的实验结果。执行如下命令：
    ```shell
    $ bash finetune_stage.sh
    ```
   这个阶段每个模型训练 3个 epoch，从一开始就开启 EMA 每隔 500 步评估保存一次模型，并且采用7折，7折，5折交叉验证
   （没有采用十折或者折数相同是因为机器性能较差，没有足够的时间）。
   
### 4. 模型融合
同样在 code 目录，执行如下命令：
```shell
$ bash test.sh
```
加载训练过程第二阶段的最佳模型，最终 b 榜得分 0.909772。  

### 5. 主要提点技术
本方案中主要有三个提分点：
1. 预处理，进行了数据增光，增加了预训练模型的样本
2.在预训练阶段优化mask方法，并且为了提供速度和精度采用了混合精度计算的方法（提高速度）和对抗训练的方法（提高精度）
3. 在每个模型的预测阶段，预测了b榜中的原始数据和对偶数据（将b榜中的样本的两句话互换位置），并进行了平均
4.采用了模型融合的方法，将三种模型的结果按照不同比例进行了加权平均，从而得到最终结果


##模型链接
nezha-cn-base链接：https://github.com/lonePatient/NeZha_Chinese_PyTorch
bert-base-chinese链接：https://blog.csdn.net/sdaujz/article/details/107547503