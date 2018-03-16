#encoding=utf-8
from sklearn.datasets.base import Bunch
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

#读取文件
def _readfile(path):
    with open(path,"rb") as fp:
        content=fp.read()
    return content

#读取ｂｕｎｃｈ对象
def _readbunchobj(path):
    with open(path,"rb") as fp:
        bunch=pickle.load(fp)
    print("%s"%bunch)
    return bunch

#写入ｂｕｎｃｈ对象
def _writebunchobj(path,bunchobj):
    with open(path,"wb") as fp:
        pickle.dump(bunchobj,fp,0)

def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path):

    stpwrdlst=_readfile(stopword_path).splitlines()
    bunch=_readbunchobj(bunch_path)
    tfidfspace=Bunch(target_name=bunch.target_name,label=bunch.label,filenames=bunch.filenames,
                     tdm=[],vocabulary={})

    #导入训练集的tf-idf词向量空间
    trainbunch=_readbunchobj(train_tfidf_path)
    tfidfspace.vocabulary=trainbunch.vocabulary

    vectorizer=TfidfVectorizer(stop_words=stpwrdlst,sublinear_tf=True,max_df=0.5,
                               vocabulary=trainbunch.vocabulary)
    tfidfspace.tdm=vectorizer.fit_transform(bunch.contents)
    _writebunchobj(space_path,tfidfspace)

    print("tf-idf词向量空间实例创建成功！！！")


if __name__=="__main__":
    stopword_path="train_word_bag/hlt_stop_word.txt"
    bunch_path="test_word_bag/test_set.dat"
    space_path="test_word_bag/testspace.dat"
    train_tfidf_path="train_word_bag/tfidfspace.dat"
    vector_space(stopword_path,bunch_path,space_path,train_tfidf_path)


























