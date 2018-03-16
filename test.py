# # import os
# # os.chdir('/usr/local/bin/libsvm-322/python')
# # from svmutil import *
# # y, x = svm_read_problem('../heart_scale')
# # m = svm_train(y[:200], x[:200], '-c 4')
# # p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
# #encoding=utf-8
# import jieba
# #将数据读出之后，放入字典中去
# tf_eng={}
# for line in open('eng.txt'):
#     seg_list=jieba.cut(line,cut_all=False) # 精确模式
#     for i in seg_list:
#         tf_eng[i]=1
#
# tf_food={}
# for line in open('food.txt'):
#     seg_list=jieba.cut(line,cut_all=False) # 精确模式
#     for i in seg_list:
#         tf_food[i]=1
#
#
# tf_tmt={}
# for line in open('tmt.txt'):
#     seg_list=jieba.cut(line,cut_all=False) # 精确模式
#     for i in seg_list:
#         tf_tmt[i]=1
#
#
# print(tf_eng)
# print(tf_food)
# print(tf_tmt)
# print(len(tf_eng))
# print(len(tf_food))
# print(len(tf_tmt))
#encoding=utf-8
import os
import jieba

#save file
def savefile(savepath,content):
    with open(savepath,'wb') as fp:
        fp.write(content.encode())


#read file
def readfile(path):
    with open(path,'r') as fp:
        content=fp.read()
    return content

def corpus_segment(corpus_path,seg_path):
    '''
    corpus_path是未分词语料库路径
    seg_path是分词后语料库存储路径
    ''''''
    '''
    catelist=os.listdir(corpus_path) #get corpus_path all childpath
    '''
    其中子目录的名字就是类别名，例如： 
    train_corpus/art/21.txt中，'train_corpus/'是corpus_path，'art'是catelist中的一个成员 
    '''
    for mydir in catelist:
        class_path=corpus_path+mydir+'/' # 拼出分类子目录的路径如：train_corpus/art/
        seg_dir=seg_path+mydir+'/'   # 拼出分词后存贮的对应目录路径如：train_corpus_seg/art/


        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)


        file_list=os.listdir(class_path) #获取子目录下的文档

        for file_path in file_list:
            fullname=class_path+file_path
            content=readfile(fullname)
            content = content.replace("\r\n", "")  # 删除换行
            content = content.replace(" ", "")  # 删除空行、多余的空格
            #使用结巴的默认分词
            content_seg=jieba.cut(content)
            savefile(seg_dir+file_path," ".join(content_seg))


    print("中文语料分词结束！！！")


if __name__=='__main__':
    corpus_path="train_corpus/"
    seg_path="train_corpus_seg/"
    corpus_segment(corpus_path,seg_path)


    corpus_path='test_corpus/'
    seg_path='test_corpus_seg/'
    corpus_segment(corpus_path,seg_path)






















