#encoding=utf-8
import os
import pickle
from sklearn.datasets.base import Bunch


def _readfile(path):
    with open(path,'rb') as fp:
        content=fp.read()

    return content

def corpus2Bunch(wordbag_path,seg_path):
    catelist=os.listdir(seg_path)#获取seg_path下的所有子目录
    #创建一个Ｂｕｎｓｈ实例
    bunch=Bunch(target_name=[],label=[],filenames=[],contents=[])
    bunch.target_name.extend(catelist)
    ''''' 
    extend(addlist)是python list中的函数，意思是用新的list（addlist）去扩充 
    原来的list 
    '''
    #获取每个目录下所有文件
    for mydir in catelist:
        class_path=seg_path+mydir+'/'
        file_list=os.listdir(class_path)
        for file_path in file_list:
            fullname=class_path+file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(_readfile(fullname))


    # print(bunch)
    #将bunch存储在wordbag_path路径下
    with open(wordbag_path,'wb') as fp:
        pickle.dump(bunch,fp,0)


    # with open(wordbag_path,"rb") as fp:
    #     pickle2=pickle.load(fp)
    #     print("pickle2:%s"%pickle2)

    print("构建文本对象结束")


if __name__=="__main__":
    #对训练集进行ｂｕｎｃｈ
    wordbag_path="train_word_bag/train_set.dat"
    seg_path="train_corpus_seg/"
    corpus2Bunch(wordbag_path,seg_path)

    #对测试集进行bunch
    wordbag_path="test_word_bag/test_set.dat"
    seg_path="test_corpus_seg/"
    corpus2Bunch(wordbag_path,seg_path)






























