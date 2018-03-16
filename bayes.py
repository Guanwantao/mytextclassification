import  pickle
from  sklearn.naive_bayes import MultinomialNB

#读取bunch对象
def _readbunchobj(path):
    with open(path,"rb") as fp:
        bunch=pickle.load(fp)
    return bunch

#导入训练集
trainpath="train_word_bag/tfidfspace.dat"
train_set=_readbunchobj(trainpath)

#导入测试集
testpath="test_word_bag/testspace.dat"
test_set=_readbunchobj(testpath)

#训练分类器：输入词袋向量和分类标签，alpha越小，迭代次数也多，越准确
clf=MultinomialNB(alpha=0.0001).fit(train_set.tdm,train_set.label)
# print(clf)
# print(test_set.tdm)
# print(train_set.label)
# print(train_set.tdm)

#预测分类结果
predicted=clf.predict(test_set.tdm)

for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)

tmtCount=0
foodCount=0
engCount=0
for atype in predicted:
    if atype=="tmt":
        tmtCount=tmtCount+1
    elif atype=="food":
        foodCount=foodCount+1
    elif atype=="eng":
        engCount=engCount+1

print("tmt"+str(tmtCount)+"\tfood"+str(foodCount)+"\teng"+str(engCount))
print("tmt:{0:.3f}".format(tmtCount/127))
print("food:{0:.3f}".format(foodCount/127))
print("eng:{0:.3f}".format(engCount/127))
print("预测完毕!!!")




















