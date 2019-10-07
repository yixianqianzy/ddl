function [count,tf,idf,weight]=tfidf(docs,term)
%docs--input documents��cell��
%term-- keywordsҲ������������ȡ,cell��
 
 
%output:count--��Ÿ����ؼ��ʳ��ֵ�Ƶ���������ĵ���
%       wordnum--����ĵ��ܵĴʻ���
 
%��������
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
    word=doc(k:tabnum(j)-1);%����������һ���� 
    Lw=length(word);
    fword=doc((tabnum(Ltab)+1):length(doc));%���һ����
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
 
%IDF---Inverse document frequency,���㹫ʽlog(N/df(j))
%      N--�ĵ�������Ŀ
%      df(j)--��ʾ����feature(j)���ĵ���Ŀ
%TF---term frequency��feature(j)���㹫ʽ���ĵ�docs(i)�еĳ��ֵ�Ƶ��Ϊcount,
%                     docs(i)���ܵĵ�����ĿΪsumw
%                      tf(i,j)=count/sumw.
Numdocs=Ldocs;
%����df
for i=1:Lterm
    tt=find(count(:,i)==0);
    df(i)=Numdocs-length(tt);
end
%����IDF
idf=log(Numdocs./df+0.5);
%����TF
for i=1:Ldocs
    tf(i,:)=count(i,:)./wordnum(i);
    weight(i,:)=100*tf(i,:).*idf;
end
 
 
 
 
 
 