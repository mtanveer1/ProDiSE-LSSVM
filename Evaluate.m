function acc = Evaluate(alpha,b,Train_data,test_label,test_X,kerf, S)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Predict and output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
obs=Train_data(:,end);
% A = Train_data(obs==1,1:end-1);
% B = Train_data(obs~=1,1:end-1);
w=alpha;
tst_num=size(test_X,1);
% C=[A;B];
T_mat=kernelfun(test_X,kerf,Train_data(S,1:end-1));
y1=(T_mat*w+b*ones(tst_num,1));

predict_Y=sign(y1);
err=sum(predict_Y~=test_label);
acc=(1-err/tst_num)*100;
end