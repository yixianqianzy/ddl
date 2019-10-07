
clear

S = pwd;
addpath(genpath(S))
load rating; % user-item interactioninfo
load content; %content info of items
% 
rand('state',0);
perm_idx1 = randperm(size(rating,1)); %user number
rand('state',0);
perm_idx2 = randperm(size(rating,2)); %item number

user=perm_idx1(1:10000);
item=perm_idx2(1:10000);
rating=rating(user,item);
content=content(item,:);

optisize=100; 
opts.batchsize = optisize;
opts.numepochs = 50;
% opts.momentum = 0;
% opts.alpha     =   1;
sp=0.1; % sparsity level: 10%
r=30; % hash code dimension 
alpha=0.0002*10^-1;
beta=0.001*10^-1;
lammda=0.001; 
k=30; % top-k ranking list
times = 30; 
[cold_item,cold_rating,row,content,Train,Test]=divide_data(rating,content,optisize,sp);
del = all(content==0,1);
content(:,del) = [];
[a,b]=size(content);
m=max(content);
content=content./(ones(a,1)*m);
sizes = [b 200];
cold_item(:, del) = [];

% [B, D, nn]= train_sae(content,opts, sizes, Train, r, alpha, beta, lammda, times); 
[B, D, nn]= train_sae(S, content,opts, sizes, Train, Test, cold_item, row,cold_rating, r, alpha, beta, lammda, times);
[hit_sp,mrr_sp] = predict_sp(k, B, D, Test) % testing for sparsity setting(10%)
[hit_cd, mrr_cd] = predict_cd(nn,cold_item, row, k,cold_rating, B) % testing for cold-start setting
save([S,'\Bd1'],'B');
save([S,'\Dd1'],'D');
save([S,'\nnd'],'nn');

fid=fopen([S,'\sp11d1_acc_fin.txt'],'a');
    fprintf(fid,'%f ',mrr_sp);
    fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_sp(i));
    end
    fprintf(fid,'\n');
    fclose(fid);
    
    fid=fopen([S,'\cd11d1_acc_fin.txt'],'a');
    fprintf(fid,'%f ',mrr_cd);
    fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_cd(i));
    end
    fprintf(fid,'\n');
    fclose(fid);










