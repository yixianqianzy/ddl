function [hit_cd, mrr_cd]=predict_cd(nn,cold_item, row, m,cold_rating, B)
nn = nnf1(nn, cold_item);
Dc=nn.a{nn.n}';
[N,~]=size(cold_rating);
% np=length(row);
s=[];
nk=zeros(N,m);
mrc=zeros(N,1);
for i=1:N
    pre_sc=B(:,i)'*Dc; %1*m
    posi=find(cold_rating(i,:)>3/5);  % postive ratings
    s(i)=length(posi);
    negc0=find(cold_rating(i,:)<4/5);
    if length(negc0)>1000
        ff=randperm(length(negc0),1000);
        negc=negc0(ff);
    else
        negc=negc0;
    end
    if s(i)~=0
        for j=1:s(i)
            pre_pair=[pre_sc(negc),pre_sc(posi(j))];
            [pre_c,~]=sort(pre_pair,'descend');
            mrc0=find(pre_c==pre_sc(posi(j)));
            mrc(i)=mrc(i)+ (1/min(mrc0));
            for kl=1:m
                if pre_c(kl)<=pre_sc(posi(j));
                    nk(i,kl)=nk(i,kl)+1;
                end
            end
        end
    end
end
hit_cd=sum(nk)/sum(s);
mrr_cd=sum(mrc)/sum(s);

end