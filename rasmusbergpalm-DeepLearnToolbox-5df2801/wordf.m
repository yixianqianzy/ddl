
clc
clear 
article=fileread('Winning - Jack Welch; Suzy Welch.txt');%读取txt文本
article=regexprep(article,'\W',' ');%将所有标点符号转换为空格 
words=regexp(lower(article),' ','split')';%转换大小写，并用文本分隔为cell
[val,idxC, idxV] = unique(words);%去除重复 val去除重复后单词，val=words(idxC),wrods=val(idxV)
n = accumarray(idxV,1);%对单词进行累加，统计出现次数
y = [val num2cell(n)];       % 未排序
[~, so]= sort(n,'descend');     % 按出现频次降序排列，so是val里单词所在的位置
words= val(so);    % 单词按降序排列
freq= n(so);           % 频次按降序排列
results = [words num2cell(freq)];    % 单词和对应的频次
xlswrite('results',results);%输出为excel文件