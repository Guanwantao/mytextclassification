from sklearn import svm
import pickle

def _readBunchobj(path):
    with open(path,"rb") as fp:
        bunch=pickle.load(fp)

    return bunch

train_path="train_word_bag/tfidfspace.dat"
train_set=_readBunchobj(train_path)

test_path="test_word_bag/testspace.dat"
test_set=_readBunchobj(test_path)

clf=svm.SVC()
clf.fit(train_set.tdm,train_set.label)
print(clf.predict(test_set.tdm))



# clf=svm.SVR()
# print(test_set.tdm)
# print(train_set.label)
# print(train_set.tdm)
# clf=clf.fit(train_set.tdm,train_set.label)
#
# result=clf.predict(test_set.tdm)
#
# print(result)








