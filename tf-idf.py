#encoding=utf-8
from sklearn.datasets.base import Bunch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def _readfile(path):
    with open(path,"rb") as fp:
        content=fp.read().decode()
    # print("content:--"+content)
    return content

def _readbunchobj(path):
    with open(path,"rb") as fp:
        bunch=pickle.load(fp)

    return bunch

def _writebunchobj(path, bunchobj):
    with open(path,"wb") as fp:
        pickle.dump(bunchobj, fp)
    print("----%s"%bunchobj)

def vector_space(stopword_path,bunch_path,space_path):

    stpwrdlst=_readfile(stopword_path).splitlines()
    bunch=_readbunchobj(bunch_path)
    #构建tf-idf词向量空间对象
    tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,
                     tdm=[],vocabulary={})


    vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5)

    # 此时tdm里面存储的就是tf-idf权值矩阵
    # vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵
    tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
    tfidfspace.vocabulary=vectorizer.vocabulary_

    _writebunchobj(space_path,tfidfspace)
    print("if-idf词向量空间实例创建成功！")


if __name__=="__main__":
    #停用词表的路径
    stopword_path="train_word_bag/hlt_stop_word.txt"
    #导入训练集bunch的路径
    bunch_path="train_word_bag/train_set.dat"
    #词向量空间保存路径
    space_path="train_word_bag/tfidfspace.dat"

    vector_space(stopword_path,bunch_path,space_path)
























