#!/usr/bin/env python
#TEAM MEMBERS: APURVA GUPTA, SURBHI PAITHANKAR, HASIKA MAHATTA

import sys
import random
import numpy as np
import math
import ast

#Function to initialize weights for all images.
def initializations(newtraindata):
    weights = []
    for k in range(0,len(newtraindata)):
        weights.append(1.0/len(newtraindata))
    return weights

#Create random pairs of decision stumps
def create_stumps():
    stumps = []
    for i in range(0,stump_length):
        x = random.randint(0,192-1)
        y = random.randint(0,192-1)
        stumps.append([x,y])
    return stumps

#Adaboost function  for creating hypothesis and corresponding weights.
def adaboost(positiveclass,negativeclass):
    stumps = create_stumps()
    z =[]
    newtraindata = []
    for k in range(0,len(traindata)):
        if traindata[k][1] in (positiveclass,negativeclass):
            newtraindata.append(traindata[k])
    weights = initializations(newtraindata)

    h ={}
    for k in range(0,stump_length):
        h[k] = {}
        h[k]['<']=""
        h[k]['>']=""

    for k in range(0,stump_length):
        val1 = stumps[k][0]
        val2 = stumps[k][1]
        traincol1 = []
        traincol2 = []
        traincol1_pos=0
        traincol2_pos=0
        traincol1_neg=0
        traincol2_neg=0

        for i in range(0,len(newtraindata)):
            #check if pixel value a is greater than pixel value at b.
            if newtraindata[i][2][val1]> newtraindata[i][2][val2]:
                traincol1.append(i)
                if newtraindata[i][1] == positiveclass:
                    traincol1_pos +=1
                else:
                    traincol1_neg+=1
            else:
                traincol2.append(i)
                if newtraindata[i][1] == positiveclass:
                    traincol2_pos +=1
                else:
                    traincol2_neg+=1
        error=0.0
        for i in range(0,len(newtraindata)):
            if  newtraindata[i][2][val1]> newtraindata[i][2][val2]:
                if traincol1_pos > traincol1_neg:
                    h[k]['>']= positiveclass
                    h[k]['<']= negativeclass
                else:
                    h[k]['<']= positiveclass
                    h[k]['>']= negativeclass

            else:
                if traincol2_pos > traincol2_neg:
                    h[k]['>']= positiveclass
                    h[k]['<']= negativeclass
                else:
                    h[k]['<']= positiveclass
                    h[k]['>']= negativeclass


        for i in range(0,len(newtraindata)):
            if  newtraindata[i][2][val1]> newtraindata[i][2][val2]:
                ans = h[k]['>']
            else:
                ans = h[k]['<']
            if ans != newtraindata[i][1]:
                error = error+weights[i]

        for i in range(0,len(newtraindata)):
            if  newtraindata[i][2][val1]> newtraindata[i][2][val2]:
                ans = h[k]['>']

            else:
                ans = h[k]['<']
            if ans == newtraindata[i][1]:
                weights[i] = weights[i] * error/(1-error)
        denom =0.0
        for i in range(0,len(newtraindata)):
            denom +=weights[i]

        for i in range(0,len(newtraindata)):
            weights[i]/= denom
        val = math.log((1-error)/error)
        z.append(val)

    return h,z,stumps

#Multiclass classfier for adaboost that uses combination of binary classifiers.
def multiclass_adaboost_train(outputfile):
    print "Running adaboost for",stump_length,"weak Classifiers."
    model = []
    h0,z0,stumps0 = adaboost('0','90')
    h1,z1,stumps1 = adaboost('0','180')
    h2,z2,stumps2 = adaboost('0','270')
    h3,z3,stumps3 = adaboost('90','180')
    h4,z4,stumps4 = adaboost('90','270')
    h5,z5,stumps5 = adaboost('180','270')
    f = open(outputfile,'w')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h0[k]),str(z0[k]),str(stumps0[k]) ))
    f.write('\n')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h1[k]),str(z1[k]),str(stumps1[k]) ))
    f.write('\n')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h2[k]),str(z2[k]),str(stumps2[k]) ))
    f.write('\n')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h3[k]),str(z3[k]),str(stumps3[k]) ))
    f.write('\n')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h4[k]),str(z4[k]),str(stumps4[k]) ))
    f.write('\n')
    for k in range(0,stump_length):
        f.write("{}\t{}\t{}\t".format( str(h5[k]),str(z5[k]),str(stumps5[k]) ))
    print "Model written into:",outputfile
    f.close()

#Function to classify test image using adaboost
def classify(positive,negative,h,z,stumps,i,stump_length):
    result = {positive:[],negative:[]}
    for k in range(0,stump_length):
        val1 = stumps[k][0]
        val2 = stumps[k][1]
        if testdata[i][2][val1]>testdata[i][2][val2]:
            ans = h[k]['>']
        else:
            ans = h[k]['<']
        result[ans].append(k)
    vote1 = 0.0
    vote2 = 0.0
    for each in result[positive]:
        vote1+= z[each]
    for each in result[negative]:
        vote2+= z[each]
    if vote1>vote2:
        return positive
    else:
        return negative

def adaboost_classifier(modelfile,outputfile):
    with open(modelfile,"r") as myfile:
        mdata= myfile.read()
    mdata = mdata.split('\n')
    modeldata = [x.strip().split("\t") for x in mdata]
    print "Model file loaded."

    h = []
    z = []
    stumps = []

    for i in range(0,6):
        str1 = ""
        str2 = ""
        str3 = ""
        j = 0
        while(j<len(modeldata[i])):
            str1+=modeldata[i][j]
            str1+=','
            j+=3
        k = 1
        while(k<len(modeldata[i])):
            str2+=modeldata[i][k]
            str2+=','
            k+=3
        m = 2
        while(m<len(modeldata[i])):
            str3+=modeldata[i][m]
            str3+=','
            m+=3

        h.append(ast.literal_eval(str1))
        z0 = str2.split(',')
        z0 = [float(x) for x in z0 if x!='']
        z.append(z0)
        str3 = str3.replace('[','').split('],')
        stumps0= [map(int, s.replace(']','').split(',')) for s in str3 if s!='']
        stumps.append(stumps0)
    stump_length= len(h[0])
    correct = 0.0
    results = {'0':0,'90':0,'180':0,'270':0}
    j =0
    f = open(outputfile,"w")
    for i in range(0,len(testdata)):
        mylist=[]
        results = {'0':0,'90':0,'180':0,'270':0}
        answer0 = classify('0','90',h[0],z[0],stumps[0],i,stump_length)
        answer1 = classify('0','180',h[1],z[1],stumps[1],i,stump_length)
        answer2 = classify('0','270',h[2],z[2],stumps[2],i,stump_length)
        answer3 = classify('90','180',h[3],z[3],stumps[3],i,stump_length)
        answer4 = classify('90','270',h[4],z[4],stumps[4],i,stump_length)
        answer5 = classify('180','270',h[5],z[5],stumps[5],i,stump_length)
        results[answer0]+=1
        results[answer1]+=1
        results[answer2]+=1
        results[answer3]+=1
        results[answer4]+=1
        results[answer5]+=1

        final= max(results.iterkeys(), key=(lambda key: results[key]))
        answer = testdata[i][1]
        if final==answer:
            correct+=1
        f.write(str(testdata[i][0]))
        f.write(" ")
        f.write(str(final))
        f.write("\n")
    f.close()
    print "-----"*18
    print "Test images with their predicted orientations have been written into",outputfile,"."
    print "Predicted orientation of",int(correct),"images correctly out of",len(testdata),"test images."
    print "-----"*18
    print "Accuracy:",correct/len(testdata)*100,"%"
    print "-----"*18

#Function to initialize weight matrix for neural nets
def init_matrix(mat, row, column):
    mat =[]
    for i in range(row):
        arr =[]
        for j in range(column):
            val = random.uniform(-1.0,1.0)
            arr.append(val)
        mat.append(arr)
    return mat

def initialize_weights():
    w1 = []
    w2 = []
    w1 = init_matrix(w1,no_of_features,hidden_nodes)
    w2 = init_matrix(w2,hidden_nodes,output_nodes)
    w1 = np.array(w1)
    w2 = np.array(w2)
    return w1,w2

#Function to return value obtained after applying sigmoid function
def sigmoid_function(z):
    z =  1.0/(1.0+np.exp(np.negative(z)))
    return z

#finds differentiation of sigmoid function value
def diff_sigmoid_function(z):
     a = (z*(1-z))
     return a

#Forward propogation
def feedforward(w1,w2,x):
    z2 = np.dot(x,w1)
    a1 = sigmoid_function(z2)
    z3 = np.dot(a1,w2)
    a2 = sigmoid_function(z3)
    return a1,a2,z3,z2

#Neural net function performing forward & backward propogation and returns weights
def neural_nets(newtrain,feature_list):
 w1,w2 = initialize_weights()
 y = actual_output(newtrain)
 for j in range(0,100):

    for i in range(0,len(newtrain)):
        x = feature_list[i]
        x = np.array(x)[np.newaxis]
        a1,a2,z3,z2 = feedforward(w1,w2,x)
        delta3 = np.multiply(((y[i]-a2)), diff_sigmoid_function(a2))
        djdw2 = np.dot(a1.T,delta3)
        delta2 = np.dot(delta3,w2.T) * diff_sigmoid_function(a1)
        djdw1 = np.dot(x.T,delta2)#error
        learning_rate = 0.1
        w1 = w1 + learning_rate * djdw1
        w2 = w2 + learning_rate * djdw2

 return w1,w2


def actual_output(newtrain):
    y = []
    for i in range(0,len(newtrain)):
        output = newtrain[i][1]
        if output == "0":
          y.append((1,0,0,0))
        elif output == "90":
          y.append((0,1,0,0))
        elif output == "180":
          y.append((0,0,1,0))
        elif output == "270":
          y.append((0,0,0,1))
    y = np.array(y)
    return y

def train_knn(train_file,model_file):
    traindata = []
    f = open(model_file,"w")
    with open(train_file,"r") as myfile:
      data= myfile.read()
      f.write(data)

def train_nnet(train_file,model_file):
    traindata =[]
    with open(train_file,"r") as myfile:
      data = myfile.read()
    tdata = data.split('\n')
    traindata = [x.split(' ') for x in tdata]
    for each in traindata:
        if (len(each)!=194):
            traindata.remove(each)
    newtrain =[]
    mylist = []
    mylist1 =[]
    feature_list = []

    for everyentry in traindata:
      mylist.append(everyentry[0])
      mylist.append(everyentry[1])
      for each in range(2,len(everyentry)):
         mylist1.append((float(int(everyentry[each]))/255.0))
      mylist.append(mylist1)
      feature_list.append(mylist1)
      newtrain.append(mylist)
      mylist = []
      mylist1 = []

    w1,w2 = neural_nets(newtrain,feature_list)
    f = open(model_file,"w")
    for each in w1:
      for every in each:
        f.write(str(every)+" ")
      f.write("\n")
    f.write("\n")
    for each in w2:
      for every in each:
        f.write(str(every)+" ")
      f.write("\n")
    f.close()
    print "Model file generated."

#KNN algorithm using manhattan distance for classification
def test_knn(test_file,model_file):
  traindata = []
  testdata = []
  with open(model_file,"r") as myfile:
    data= myfile.read()
  tdata = data.split('\n')
  print "Model file loaded."
  with open(test_file,"r") as myfile:
    data= myfile.read()
  ndata = data.split('\n')

  traindata = [x.split(' ') for x in tdata]
  testdata =  [x.split(' ') for x in ndata]
  
  for each in traindata:
    if len(each)!=194:
      traindata.remove(each)

  for each in testdata:
    if len(each)!=194:
      testdata.remove(each)
  newtrain = []
  mylist = []
  newtest = []
  mylist1=[]
  for everyentry  in traindata:
    mylist.append(everyentry[0])
    mylist.append(everyentry[1])
    for each in range(2,len(everyentry)):
        mylist1.append(int(everyentry[each]))
    mylist.append(mylist1)
    newtrain.append(mylist)
    mylist =[]
    mylist1 = []

  for everyentry  in testdata:
    mylist.append(everyentry[0])
    mylist.append(everyentry[1])
    for each in range(2,len(everyentry)):
      mylist1.append(int(everyentry[each]))
    mylist.append(mylist1)
    newtest.append(mylist)
    mylist =[]
    mylist1 = []
  total = 0
  diff_list = []
  final_answers = []

  correctans = 0
  count = 0
  f = open("output.txt","w")
  for testline in newtest:
    diff_list = []
    u = (testline[2])
    for every in newtrain:
      v = (every[2])
      dist = sum(abs(a-b) for a,b in zip(u, v))
      diff_list.append((dist,every[0],every[1]))

    diff_list.sort()
    final_list = diff_list[0:k]
    label_dir = {}
    for each in final_list:
      label_dir[each[2]] = 0

    for each in final_list:
      label_dir[each[2]] +=1

    final_label= max(label_dir.iterkeys(), key=(lambda key: label_dir[key]))
    final_answers.append(final_label)

    count+=1
    f.write(testline[0])
    f.write(" ")
    f.write(final_label)
    f.write("\n")
    if final_label==testline[1]:
      correctans+=1
  f.close()
  accuracy = 0.000
  print "-----"*18
  print "Test images with their predicted orientations have been written into output.txt."#,outputfile,"."
  print "Predicted orientation of",int(correctans),"images correctly out of",len(final_answers),"test images."
  print "-----"*18

  accuracy = float(correctans)/float(len(final_answers))
  print "accuracy=",accuracy*100,"%"

#Classification of test image using nnet.
def test_nnet(test_file,model_file):

   w1=[]
   w2=[]
   testdata =[]
   with open(test_file,"r") as myfile1:
     data1 = myfile1.read()
   ndata = data1.split('\n')
   testdata= [x.split(' ') for x in ndata]
   for each in testdata:
     if (len(each)!=194):
        testdata.remove(each)
   with open(model_file,"r") as mfile:
        data = mfile.read()
   data1 = data.split('\n\n')
   print "Model file loaded."

   w2_data = data1[1].split('\n')
   w2 = [x.split(' ') for x in w2_data if x!='']
   for each in w2:
       for every in each:
           if every=='':
               each.remove(every)
   w1_data= data1[0].split('\n')
   w1 = [x.split(' ') for x in w1_data if x!='']
   for each in w1:
       for every in each:
           if every=='':
               each.remove(every)
   mylist=[]
   mylist1=[]
   feature_list1=[]
   newtest=[]
   for everyentry in testdata:
    mylist.append(everyentry[0])
    mylist.append(everyentry[1])
    for each in range(2,len(everyentry)):
        mylist1.append((float(int(everyentry[each]))/255.0))
    mylist.append(mylist1)
    feature_list1.append(mylist1)
    newtest.append(mylist)
    mylist = []
    mylist1=[]
   predicted_array=[]
   label = ['0','90','180','270']
   w1 = np.array(w1,dtype=float)
   w2 = np.array(w2,dtype=float)
   for i in range(0,len(newtest)):
     x = np.array(feature_list1[i])
     z2 = np.dot(x,w1)
     a2 = sigmoid_function(z2)
     z3 = np.dot(a2,w2)
     yhat = sigmoid_function(z3)
     max = -99.99
     max_label = 0
     for j in range(0,len(yhat)):

        if yhat[j]>max:
            max = yhat[j]
            max_label = j
     if max_label==0:
            predicted_array.append(label[0])
     elif max_label==1:
            predicted_array.append(label[1])
     elif max_label==2:
            predicted_array.append(label[2])
     elif max_label==3:
            predicted_array.append(label[3])
   correct= 0.0
   f = open("output.txt","w")
   for i in range(0,len(newtest)):
     #print "Predicted",predicted_array[i]
     #print "Actual",newtest[i][1]
     f.write(newtest[i][0]+" ")
     f.write(predicted_array[i])
     f.write("\n")
     if predicted_array[i] == newtest[i][1]:
        correct +=1
   f.close()
   print "-----"*18
   print "Test images with their predicted orientations have been written into output.txt."#,outputfile,"."
   print "Predicted orientation of",int(correct),"images correctly out of",len(newtest),"test images."
   print "-----"*18
   accuracy = float(float(correct)/len(newtest))
   print "accuracy",accuracy*100,"%"


type = (sys.argv[1])

if type == "train":
  
  train_file = sys.argv[2]
  model_file = sys.argv[3]
  model = sys.argv[4]

  if model=="nearest":
    k = 25
    train_knn(train_file,model_file)
    print "Model file generated successfully!"

  elif model=="adaboost":
    stump_length = 500
    print "Model file will be generated in 3-4 minutes. Please wait.."
    with open(train_file,"r") as myfile:
      data = myfile.read()
    tdata = data.split('\n')
    traindata = [x.split(' ') for x in tdata]
    for each in traindata:
        if (len(each)!=194):
            traindata.remove(each)
    newtrain =[]
    mylist = []
    mylist1 =[]
    feature_list = []

    for everyentry in traindata:
      mylist.append(everyentry[0])
      mylist.append(everyentry[1])
      for each in range(2,len(everyentry)):
         mylist1.append((float(int(everyentry[each]))/255.0))
      mylist.append(mylist1)
      feature_list.append(mylist1)
      newtrain.append(mylist)
      mylist = []
      mylist1 = []
    traindata = newtrain
    multiclass_adaboost_train(model_file)
    
  elif model=="nnet" or model=="best":
      hidden_nodes = 20
      no_of_features = 192
      output_nodes = 4
      if model == 'best':
          print "We are using Neural Networks as our best algorithm."
      print "Running for one hidden layer and",hidden_nodes,"hidden nodes."
      print "Please wait for 6-7 minutes while the model file is being generated."
      train_nnet(train_file,model_file)
  else:
    print "Please enter correct model:-nearest,adaboost or nnet"

elif type=="test":

 test_file = sys.argv[2]
 model_file = sys.argv[3] 
 model = sys.argv[4]
 testdata = test_file
 if model=="nearest":
     k = 25
     print "Running KNN for K=25"
     print "Please wait for 10-15 minutes."
     test_knn(test_file,model_file)

 elif model=="adaboost":
    with open(test_file,"r") as myfile:
      data = myfile.read()
    tdata = data.split('\n')
    testdata = [x.split(' ') for x in tdata]
    for each in testdata:
        if (len(each)!=194):
            testdata.remove(each)
    newtest =[]
    mylist = []
    mylist1 =[]
    feature_list = []

    for everyentry in testdata:
      mylist.append(everyentry[0])
      mylist.append(everyentry[1])
      for each in range(2,len(everyentry)):
         mylist1.append((float(int(everyentry[each]))/255.0))
      mylist.append(mylist1)
      feature_list.append(mylist1)
      newtest.append(mylist)
      mylist = []
      mylist1 = []
    testdata = newtest
    adaboost_classifier(model_file,"output.txt")

 elif model=="nnet" or model=="best":
     if model=='best':
         print "We are using Neural Network as our best algorithm."
     test_nnet(test_file,model_file)

 else:
    print "Please enter correct model:-nearest,adaboost or nnet"

else:
  print "Please enter either 'train' or 'test'"
