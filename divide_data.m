function [cold_item,cold_rating,row,content,Train,Test] = divide_data(rating,content,optisize,sp)
rating = rating./5;
I=rating>3/5;
cold=find(sum(I)<5);  % cold start items
cold_rating=rating(:,cold); % testing dataset for cold start setting
rating(:,cold)=[]; % training and testing dataset for sparsity settings
row=find(sum(cold_rating,2)>3/5)';  % choose users with ratings in cold start setting
cold_item=content(cold,:); %content tf-idf features for cold start items
content(cold,:)=[]; % content training dataset
[~,M]=size(rating);
rr=mod(M,optisize);
if rr~=0
    rating(:,1:rr)=[];
    content(1:rr,:)=[];
end
[N,M]=size(rating);

% randomly choose sp = 10% ratings from rating dataset as training data.
% The rest as testing data
Train = zeros(N,M);
Test = zeros(N,M);
id_nz = find(rating);
rand_id = randperm(length(id_nz));
rand('state',0);
id_train = id_nz(rand_id(1:ceil(length(id_nz)*sp)));
id_test = setdiff(id_nz,id_train);
Train(id_train)=rating(id_train);
Test(id_test)=rating(id_test);
Train=sparse(Train);
Test=sparse(Test);

%for j=1:M
%    [I,J,Val] = find(rating(:,j));
%    rand('state',0);
%    [test,train] = crossvalind('LeaveMOut',nnz(rating(:,j)),ceil(nnz(rating(:,j))*sp));
%    Train(:,j)=sparse(I(train), J(train), Val(train),N,1);
%    Test(:,j)=sparse(I(test),J(test), Val(test), N,1);
%end
end