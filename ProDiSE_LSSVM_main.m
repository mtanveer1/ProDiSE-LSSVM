clc; clear all; close all;
load('Train.txt')
load('Test.txt')
seed=[1:1:20];
[samples_train, features_train]=size(Train);
[samples_test, features_test]=size(Train);
samples=samples_train+samples_test;
features=features_train+features_test;
m=round(0.2*samples); 
FunPara.c_1=10^-2;
FunPara.kerfpara.pars=100;
FunPara.kerfpara.type='rbf';


[alpha, b, S, time] =  ProDiSE_LSSVM_Train(Train(:,1:end-1), Train(:,end),  FunPara.kerfpara, FunPara.c_1, m, seed);
accuracy=Evaluate(alpha,b,Train,Test(:,end),Test(:,1:end-1),FunPara.kerfpara, S);

accuracy