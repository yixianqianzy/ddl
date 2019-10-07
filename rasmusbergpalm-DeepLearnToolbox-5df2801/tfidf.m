function [count,tf,idf,weight]=tfidf(docs,term)
%docs--input documents，cell型
%term-- keywords也就是特征词提取,cell型
 
 
%output:count--存放各个关键词出现的频率在整个文档中
%       wordnum--存放文档总的词汇数
 
%测试用例
%*****************************************************************
%clear all
doc1='www washingtonpost com wp-adv mediacenter images wpni skin2 jpg';
doc2='www washingtonpost com wp-adv mediacenter images about us welcome gif';
doc3='media washingtonpost com wp-adv mediacenter images wpni mediakit hdr top gif';
doc4='www washingtonpost com wp-adv mediacenter html research demographics html';
 
docs={
doc1,doc2,doc3,doc4
};
term={
'washingtonpost','mediacenter','images','media','wpni','html','jpg','gif'
};
%%*************************************************************************
 
Ldocs=length(docs);
Lterm=length(term);
tf=zeros(Ldocs,Lterm);
idf=zeros(1,Lterm);
count=zeros(Ldocs,Lterm);
wordnum=[];
weight=zeros(Ldocs,Lterm);
p=' ';
i=1;
for i=1:Ldocs
    doc=cell2mat(docs(i));
    tabnum=find(doc==p);
    Ltab=length(tabnum);
    wordnum(i)=Ltab+1;
    k=1;
    for j=1:Ltab
    word=doc(k:tabnum(j)-1);%会少输出最后一个词 
    Lw=length(word);
    fword=doc((tabnum(Ltab)+1):length(doc));%最后一个词
    Lfw=length(fword);
        for jj=1:Lterm
            aterm=cell2mat(term(jj));
            Lat=length(aterm);
            if Lat==Lw||Lat==Lfw
                if strcmpi(word,aterm);
                    count(i,jj)=count(i,jj)+1;
                    if jj<6
                        count(i,jj)=count(i,jj)+3;
                    end
                end
            end
        end
     k=tabnum(j)+1;
    end
end
 
%IDF---Inverse document frequency,计算公式log(N/df(j))
%      N--文档的总数目
%      df(j)--表示包含feature(j)的文档数目
%TF---term frequency，feature(j)计算公式在文档docs(i)中的出现的频数为count,
%                     docs(i)的总的单词数目为sumw
%                      tf(i,j)=count/sumw.
Numdocs=Ldocs;
%计算df
for i=1:Lterm
    tt=find(count(:,i)==0);
    df(i)=Numdocs-length(tt);
end
%计算IDF
idf=log(Numdocs./df+0.5);
%计算TF
for i=1:Ldocs
    tf(i,:)=count(i,:)./wordnum(i);
    weight(i,:)=100*tf(i,:).*idf;
end
 
 
 
 
 
 