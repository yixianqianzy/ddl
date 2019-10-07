function [output_matrix] = tfidf(data, method)
% input: a matrix where all elements are measured as term frequency
% output: depended on the input method
% method: tfidf
% data: [i*j] matrix represents i-th document and the j-th term
% n_nuid: the number of documents where appear a term j at least one time
%         ex: n_nuid(6)=9 represents there are 9 documents appear the 6-th term
%         a vector of which size is based on the size of vocabulary
% n_fulld:the number of documents where appear a term j
%         ex: n_fulld(6)=9 means there may be <=9 documents appear the 6-th term
% n_term_d: sum of number of occurrences of all terms in a document
%         the length of a document. ex: a document i has 'a', 'a', 'b', 'c', return 4
%         the size of the vector is the same as the number of documents
% n_avg_term_d: average number of terms in a document     
 
 
[n_nuid, n_fulld, n_term_d, n_avg_term_d] = estimate_n(data);
output_matrix=zeros(size(data));
temp_matrix=zeros(size(data));
 
for i=1:size(data,1) %the i-th row
    for j=1:size(data,2) %the j-th column
        switch method
            case 'tfidf' %tf*log(N/n)
                output_matrix(i,j)=data(i,j)/n_term_d(i)*log((size(data,1)/n_nuid(j)));
            end %ending switch
            if isnan(output_matrix(i,j))
                output_matrix(i,j)=0;
            end
    end
end
 
function [n_unid, n_fulld,n_term_d, n_avg_term_d] = estimate_n(data)
    %n_unid: number of unique document appear a term
    %n_fulld: number of token  document appear a term
    %n_term_d: number of token terms in a document
    %n_avg_term_d: average number of token terms in a document
    %n_avgterm: average number of document appear a term
     
    for j=1:size(data,2)%the j-th column
        temp_n1=0;
        temp_n2=0;
        for i=1:size(data,1) %the i-th row
            if data(i,j)>=1
                temp_n1=temp_n1+1;
            end
            temp_n2=temp_n2+data(i,j);
        end 
        n_unid(j)=temp_n1;
        n_fulld(j)=temp_n2;
    end
     
    %sum of number of occurrences of all terms in a document
    temp_n_term=0;
    for i=1:size(data,1)        
        for j=1:size(data,2)
            temp_n_term = temp_n_term+data(i,j);
        end
        n_term_d(i)=temp_n_term;
        temp_n_term=0;
    end
    n_avg_term_d=mean(n_term_d);
end
end