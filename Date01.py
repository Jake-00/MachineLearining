from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def datasets_demo():
    '''
    sklearn数据集使用
    :return:
    '''

    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集：\n", iris)
    print("查看数据集描述\n", iris["DESCR"])
    print("查看特征名字：\n", iris.feature_names)
    print("查看特征值：\n", iris.data, iris.data.shape)

    # 数据集划分，test_size=0.2 意思是测试集占数据集的百分之20，设置 random_state 是为了检验同样的随机条件下，那个算法好
    x_train, x_test, y_trian, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值：\n", x_train, x_train.shape)

    return None


def dict_deomo():
    '''
    字典特征提取
    :return:
    '''
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]
    # 1.实例化一个转换器类
    transfer = DictVectorizer(sparse=False)            # 默认是 sparse=True ,返回sparse矩阵（稀疏矩阵）
    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)
    print("特征名字：\n", transfer.get_feature_names())
    # 3.
    return None


def count_demo():
    '''
    文本特征提取：CountVectorizer
    :return:
    '''
    data = ["life is short,i like like python",
            "life is too long,i dislike python"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer(stop_words=['is', 'too'])
    # 2.调用fit_transform()方法
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())       # CountVectorizer() 没有sparse参数,若想看hot-code就要调用 .toarray() 方法
    print("data_new's type:", type(data_new))
    print("特征名字：\n", transfer.get_feature_names())

    return None


def count_chinese_demo():
    '''
    中文文本特征提取：CountVectorizer
    :return:
    '''
    data = ["我爱北京天安门",
            "天安门上太阳升"]
    # 1.实例化一个转换器类
    transfer = CountVectorizer()
    # 2.调用fit_transform()方法
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new.toarray())       # CountVectorizer() 没有sparse参数,若想看hot-code就要调用 .toarray() 方法
    print("data_new's type:", type(data_new))
    print("特征名字：\n", transfer.get_feature_names())

    return None


def cut_words(text):
    '''
    进行中文分词 "我爱北京天安门" --> "我 爱 北京 天安门"
    :param text:
    :return:
    '''
    text = " ".join(list(jieba.cut(text)))
    print(text)

    return text


def count_chinese_demo2():
    '''
    中文文本特征提取，自动分词：CountVectorizer
    :return:
    '''
    data = data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
                   "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
                   "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_words(sent))
    print(data_new)

    # 1.实例化一个转换器类
    transfer = CountVectorizer(stop_words=['因为', '所以'])
    # 2.调用fit_transform()方法
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())       # CountVectorizer() 没有sparse参数,若想看hot-code就要调用 .toarray() 方法
    print("data_new's type:", type(data_final))
    print("特征名字：\n", transfer.get_feature_names())

    return None


def tfidf_demo():
    '''
    用TD-IDF的方法进行文本特征抽取
    :return:
    '''
    data = data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
                   "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
                   "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    data_new = []
    for sent in data:
        data_new.append(cut_words(sent))
    print(data_new)

    # 1.实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=['因为', '所以'])
    # 2.调用fit_transform()方法
    data_final = transfer.fit_transform(data_new)
    print("data_new:\n", data_final.toarray())  # CountVectorizer() 没有sparse参数,若想看hot-code就要调用 .toarray() 方法
    print("data_new's type:", type(data_final))
    print("特征名字：\n", transfer.get_feature_names())

    return None


def minmax_scaler():
    """
    归一化
    :return:
    """

    # 1. 调用一个转换器
    data = pd.read_csv("D:\chrome_download\Python3_MachineLearing\datingTestSet2_reshape.txt")
    data = data.iloc[:, 0:3]  # .iloc[] 索引
    print("data:\n", data)

    # 2. 实例化一个转换器
    transfer = MinMaxScaler(feature_range=[0, 1])

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def stand_demo():
    """
    标准化
    标准化比归一化更好
    :return:
    """

    # 1. 调用一个转换器
    data = pd.read_csv("D:\chrome_download\Python3_MachineLearing\datingTestSet2_reshape.txt")
    data = data.iloc[:, 0:3]  # .iloc[] 索引
    # print("data:\n", data)

    # 2. 实例化一个转换器
    transfer = StandardScaler()

    # 3. 调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)


if __name__ == "__main__":
    # 代码1：skleaarn数据集使用
    # datasets_demo()
    # 代码2：字典特征提取
    # dict_deomo()
    # 代码3：文本特征提取：CountVectorizer()
    # count_demo()
    # 代码4：中文文本特征提取：CountVectorizer()
    # count_chinese_demo()
    # 代码5：中文文本特征提取，自动分词
    # count_chinese_demo2()
    # 代码6：进行中文分词 "我爱北京天安门" --> "我 爱 北京 天安门"
    # cut_words("我爱北京天安门")
    # 代码7：用TD-IDF的方法进行文本特征抽取
    # tfidf_demo()
    # 代码8：归一化
    # minmax_scaler()
    # 代码9：标准化
    stand_demo()
