import os




path1="test_corpus/test/test.txt"
f=open(path1,"rb")
count=1
for line in f.readlines():
    with open("test_corpus/test/"+str(count)+".txt","wb") as fp:
        fp.write(line)
    count=count+1
