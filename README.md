## Resources
### Dataset
- 1. Fake new dataset 1: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data

## DayLog
### 2024/04/05
- finished Batch size DataFrame handler, next step put batch dataframe to tokenzier, join them to 1 big featrue matrix


### 419:
1 需要设置batch，因为梯度是按照batch来更新的，如果没有batch，可能会导致单个训练更新梯度过大，学习效果不好

TODO:
- [x] set batch for loss function or trian data 


2 震荡问题，准确率在50左右震荡，学习止步不前
- [ ] 需要吧LSTM 网络输入层的batch也改了，要不然参数没法更新

### 420:
	a. 无法合并到一个batch中，sequence lenghtb不够长
    - [x] padding ： 缺点❌：sequence lentgh差距过大 （54 words VS 888 words）padding导致稀疏矩阵
    - [ ] padding with masks: padding 到最大长度 ex 888x300，随后把补齐的过程用mask屏蔽掉- - TORCH.NN.UTILS.RNN.PACK_PADDED_SEQUENCE() - torch.nn.rnn.pack_padded_sequence()


### 422
	a)分类问题的number/size of output应该等于number of classes，例如数据集3个类ABC，那么输出的结果应该为1x3的vector，每个值代表对应三个类的概率。
	b) 训练效果随着epoch增大而降低，开始100以内的epoch还可以实现 > 70% 准确率，epoch达到500-600左右出现大量<40% 的准确率:
￼
- [ ] 调用twitter dev API爬取 Extra Feature used by Fake news classifier（例如 post点赞/转发/评论数+account发文频率）
- [ ] 把Extra Feature used by Fake news classifier 作为额外feature嵌入到embeddings matrix中，增加判断因素
￼
￼



### 423
随机梯度下降法（是目前使用的方法，每执行一个数据就计算一次梯度、更新一次梯度，训练速度较快，但是永远不会收敛，在目标点（最优点附近晃来晃去）



### 424
- [ ] 简化程序调用，减少重复性


￼

### 501
- [ ] mini-batch 和 随机梯度下降结合使用，先随机梯度下降（加快收敛过程），后面mini-batch（确保可以稳定收敛）
- [x] 设置epoch，然后argmax(accuary)


### 503
1. 估算chunk来划分test size问题
	def datarow_count(self):
        	with open(self.args.dataset_path + self.args.dataset_name + '/WELFake_Dataset.csv') as fileobject:
            	self._total_length = sum(1 for row in fileobject)
        	logging.info(f"\n ** DataFile have {self._total_length} rows of data! \n")
        	# calculate the total chunk number
        	self._chunk_number = self._total_length // self.args.chunk_size # this is estimate, because later iterator will delete some row in every chunk

2. arg max 需要额外存储每个epoch的模型

针对问题：自动调参
- [ ] 预设多个hyper-parameters set，和epoch结合来用，最好可以随机初始化hyper parameter
- [ ] (可选) 自动化配置超参数架构，通过计算



### 505
针对问题：arg max 需要额外存储每个epoch的模型
- [ ] 设置accurary变好就保存



### 506 / 07
- [x] 日志函数改为装饰函数：减少代码复杂度s
	def log_dec(func):
		@warp(func)		
		def record_log_call():
			logging.info(“…”)
			return func
	
	@log_dec
	def main_function():
		# do something

- [x] 定义abstract 超类，把所有多利用变量放在超类中，子类使用super()__init__，避免重复声明
- [x] tokenized chunk保存在self中，把network设置为data handler的子类。
    - [x] ￼
    - [x] 问题：内存占用过高，思路：



    - [x] 解决：考试使用自定义yield迭代器？
        - [x] 把整个data_handler变成一个迭代器/生成器，每次调用仅处理args.chunk_size大小的数据，因此：
            - [x] 要把调用源放在network.py里面，当train需要数据时候现场生成chunk_size 的tokenized


### 508

- [ ] Automatic Algorithm Configuration based on Local Search - 使用自动参数初始化和参数随机选择方法
- [ ] mini-batch 和 随机梯度下降结合使用，先随机梯度下降（加快收敛过程），后面mini-batch（确保可以稳定收敛）


### 509
- [x] 尝试把data handler 封装成 class iterator，重写__next__()和StopIteration()方法
- [x] chunk调用使用__next__()方法，for循环可能存在问题❌


### 510
- [x] data_handler和data_handler_iterator解藕



### 513
- [x] handler_init()无限循环调用问题
    - [x] 所在：data handler 的__init__会循环调用构建子类iterator 的方法
- Draft更新新设计模式


- [ ] 需要找出model对应子类（实例类）的共有属性和方法

v1900
- [ ] mini-batch 合并长度时无法合并，因为每个句子长度不一致 888个词 vs 54个词
    - [ ] 网上解决方法： 设置一个 max_padding，超过则截断，不足则补齐
    - [ ] 但是我们的句子长度差距过大（888 vs 54）可能损失很多精度和信息



### 514
fix some bugs
￼



### 515
- [ ] 错误保存没有生效，log日志也没有记录




### 516
- [x] 部署SMAC3 Auto ML 框架
    - [x] 统计+筛选需要Auto optimize的超参数
    - [x] 框定每个超参数的搜索范围
- [ ] LDA Model Loading classifier model
issues
- [ ] logging missing
- [x] epoch遍历结束时，next（）还会继续返回，返回None




### 519
- [ ]     def force_save_model(self): 增加回滚支持，确保强制保存的成功运行
- [ ] 网络结构需要对SMAC3进行修改，只寻找Network的最佳网络结构



### 520
- [ ] matliplot绘制训练 batch(x) - accuary(y)曲线
- [ ] 写一个自动测试classifier 模型的算法（ex. 给100组随机新闻去检测每个模型的accuary）
- [ ] 模型加载过程中的参数问题 （autoPytorch输出不一样的模型配置文件）
- [x] UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.02901458016193781 and num_layers=1
- [x] AttributeError: 'function' object has no attribute 'dim':()  解决：错误信息表明在调用 torch.nn.init.xavier_uniform_ 函数时，传递了一个函数对象而不是一个张量。这是因为你错误地传递了 self.lstm.parameters（一个方法）而不是 self.lstm.parameters()（一个生成器返回所有参数的迭代器）。
