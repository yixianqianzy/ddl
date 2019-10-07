function [hit_sp,mrr_sp]=predict_sp(m, B, D, Test)
[N,~]=size(Test);
sw=[];
mr=zeros(N,1);
nw=zeros(N,m);
for i=1:N
    pre_sw=B(:,i)'*D;  %1*m
    cp=find(Test(i,:)>3/5); %positive ratings
    sw(i)=length(cp);
    negw0=find(Test(i,:)<4/5);
    if length(negw0)>1000
        rand('state',0);
        ffw=randperm(length(negw0),1000);
        negw=negw0(ffw);
    else
        negw=negw0;
    end
    if sw(i)~=0
        for j=1:sw(i)
            pre_pairw=[pre_sw(negw),pre_sw(cp(j))];
            [pre_w,~]=sort(pre_pairw,'descend');
            mr0=find(pre_w==pre_sw(cp(j)));
            mr(i)=mr(i)+ (1/min(mr0));
            for k=1:m
                if pre_w(k)<=pre_sw(cp(j))
                    nw(i,k)=nw(i,k)+1;
                end
            end
        end
    end
end
hit_sp=sum(nw)/sum(sw);
mrr_sp=sum(mr)/sum(sw);
end

