function [B, D, nn]= train_sae(S, content,opts, sizes, Train, Test, cold_item, row,cold_rating, r, alpha, beta, lammda, times)
[N,M]=size(Train);
% pretrain
% sae=saepretrain(sae,opts,content);
% [a,~]=size(content);
% m=max(content);
% %content=content./m;
% %  [m,~]=max(content,[],2);
%
% content=content./(ones(a,1)*m);
sae = saesetup(sizes);
sae = saetrain(sae, content, opts);

nn = nnsetup([sizes, r]);
nn.W{1} = sae.ae{1}.W{1};
% nn.W{2} = sae.ae{2}.W{1};
% nn = dbnunfoldtonn(dbn);
nn = nnf1(nn, content);
% D=rand(r,M);
% D(D>0.5)=1; D(D<=0.5)=-1;
% Initialization
D=sign(nn.a{nn.n})';
rand('state',0);
B=rand(r,N);
B(B>0.5)=1; B(B<=0.5)=-1;
rand('state',0);
X=randn(r,N);
Y=randn(r,M);


% Training
for t=1:times
    B = B_sub( Train, B, D, X,r, N, alpha); % B-sub problem
    D = D_sub( Train, B, D, Y, r, M, beta, nn.a{nn.n}, lammda); % D-sub problem
    nn.learningRate = 0.001;
    opts.numepochs =1;
    [nn,L] = nntrain(nn, content, D', opts); % fine-tune process
    nn = nnf1(nn, content); % forward process
    X = UpdateSVD(B); % update X
    Y = UpdateSVD(D); % update Y
    lr(t) = rating_loss(Train, B, D, r);
    ld(t)=deep_loss(D, nn);
k=30;
    [hit_sp,mrr_sp] = predict_sp(k, B, D, Test);
    [hit_cd, mrr_cd] = predict_cd(nn,cold_item, row, k,cold_rating, B);
    fid=fopen([S,'\sp11d1_acc.txt'],'a');
    fprintf(fid,'%f ',mrr_sp);
    fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_sp(i));
    end
    fprintf(fid,'\n');
    fclose(fid);
    
    fid=fopen([S,'\cd11d1_acc.txt'],'a');
    fprintf(fid,'%f ',mrr_cd);
    fprintf(fid,'\n');
    for i=1:k
        fprintf(fid,'%f ',hit_cd(i));
    end
    fprintf(fid,'\n');
    fclose(fid);
    
    fid=fopen([S,'\lossd1.txt'],'a');
    fprintf(fid,'%f %f %f \n',lammda, lr(t), ld(t));
    fclose(fid);
    %     figure;
% plot(L);
end
 figure;
 plot(1:times,lr);
  figure;
 plot(1:times,ld);
end

function B  = B_sub( R, B, D, X,r, N, alpha)
%B-subproblem
I=R>0;
Vu=sum(I,2);
for i=1:N
    for k=1:r
        Dd{i,k}=(D(k,:).*I(i,:)*D')'; %Rt:index matrix of ratings; Dd(i,k): rx1
        da{i,k}=D(k,:).*I(i,:)*(2*r*R(i,:)'-r);
    end
end
for i=1:N
    FLAg=1; step=0;
    while FLAg
        for k=1:r
            term1=B(:,i)'*Dd{i,k};
            term2=Vu(i)*B(k,i);
            term3=da{i,k}+alpha*X(k,i);
            bi_bar(k)=term1-term2-term3;
            if bi_bar(k)~=0
                if B(k,i)==-sign(bi_bar(k))
                    fl(k)=0;
                else
                    B(k,i)=-sign(bi_bar(k));
                    fl(k)=1;
                end
            end
        end
        FLAg=sum(fl);
        step=step+1;
    end
end
end

function  D  = D_sub( R, B, D, Y, r, M, beta, sae_out, lammda)
% D-subproblem
I=R>0;
Vt=sum(I);
for j=1:M
    for k=1:r
        Bb{j,k}=(B(k,:).*I(:,j)'*B')'; %Rt:index matrix of ratings; Dd(i,k): rx1
        ba{j,k}=B(k,:).*I(:,j)'*(2*r*R(:,j)-r);
    end
end
for j=1:M
    FLAg=1; step=0;
    while FLAg
        for k=1:r
            term1=D(:,j)'*Bb{j,k};
            term2=Vt(j)*D(k,j);
            term3=ba{j,k}-beta*Y(k,j);
            term4=lammda*sae_out(j,k)/2;
            dj_bar(k)=term1-term2-term3-term4;
            if dj_bar(k)~=0
                if D(k,j)==-sign(dj_bar(k))
                    fl(k)=0;
                else
                    D(k,j)=-sign(dj_bar(k));
                    fl(k)=1;
                end
            end
        end
        FLAg=sum(fl);
        step=step+1;
    end
end
end

function U = my_MGS(U, K)
[m,n] = size(U);
U = [U zeros(m,K-n)];
for i = n+1:K
    v = rand(m,1);
    v = v-mean(v);
    for j = 1: i-1
        v = v-(U(:,j)'*v)*U(:,j);
    end
    v = v/norm(v);
    %v = l2_normalize(v');
    U(:,i) = v';
end
end

function P=P_sub(Q, X, I, A, N, alpha, r)
for i=1:N
    QQ_t=2*(Q.*repmat(I(i,:),[r,1]))*Q';
    IA_q=2*Q*(I(i,:).*A(i,:))';
    P(:,i)=(alpha*eye(r)+QQ_t)^-1*(2*alpha*X(:,i)+IA_q);
end
end

function Q=Q_sub(P, Y, I, A, DBN_out, M, beta, lammda, r)
F=DBN_out';
for j=1:M
    PP_t=2*(P.*repmat(I(:,j),[1,r])')*P';
    IA_p=2*P*(I(:,j).*A(:,j));
    Q(:,j)=((lammda+beta)*eye(r)+PP_t)^-1*(2*beta*Y(:,j)+lammda*F(:,j)+IA_p);
end
end

function H_v = UpdateSVD(W)
%UpdateSVD: update rule in Eq.(16)
[b,n] = size(W);
m = mean(W,2);
JW = bsxfun(@minus,W,m);
JW = JW';
[P,ss] = eig(JW'*JW);
ss = diag(ss);
zeroidx = (ss <= 1e-10);
if sum(zeroidx) == 0
    H_v = sqrt(n)*P*(JW*P*diag(1./sqrt(ss)))';
else
    ss = ss(ss>1e-10);
    Q = JW*P(:,~zeroidx)*diag(1./sqrt(ss));
    Q = my_MGS(Q, b);
    H_v = sqrt(n)*P *Q';
end
end

function dbn=pretrain(dbn,opts,train_x,r)
[~,t]=size(train_x);
[m,~]=max(train_x,[],2);
train_x=train_x./(m*ones(1,t));
rand('state',0)
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
end