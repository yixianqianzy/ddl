
clc
clear 
article=fileread('Winning - Jack Welch; Suzy Welch.txt');%��ȡtxt�ı�
article=regexprep(article,'\W',' ');%�����б�����ת��Ϊ�ո� 
words=regexp(lower(article),' ','split')';%ת����Сд�������ı��ָ�Ϊcell
[val,idxC, idxV] = unique(words);%ȥ���ظ� valȥ���ظ��󵥴ʣ�val=words(idxC),wrods=val(idxV)
n = accumarray(idxV,1);%�Ե��ʽ����ۼӣ�ͳ�Ƴ��ִ���
y = [val num2cell(n)];       % δ����
[~, so]= sort(n,'descend');     % ������Ƶ�ν������У�so��val�ﵥ�����ڵ�λ��
words= val(so);    % ���ʰ���������
freq= n(so);           % Ƶ�ΰ���������
results = [words num2cell(freq)];    % ���ʺͶ�Ӧ��Ƶ��
xlswrite('results',results);%���Ϊexcel�ļ�