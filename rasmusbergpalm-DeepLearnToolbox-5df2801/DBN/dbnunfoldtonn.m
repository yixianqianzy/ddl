function nn = dbnunfoldtonn(dbn, outputsize)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added. Ŀ�������label������������MINST����10��DBNֻ����ѧϰfeature  
%   ����˵��ʼ��Weight����һ��unsupervised learning������supervised���ÿ�NN  
    if(exist('outputsize','var'))    %��⡰outputsize�� ����Ϊ���������Ƿ����
        size = [dbn.sizes outputsize];   %784*100*10
    else
        size = [dbn.sizes];  
    end
    nn = nnsetup(size);
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W]; %��DBNԤѵ���õ���ÿһ��Ȩ��+ƫ�á���NN��ÿ��Ȩ�س�ʼ��
    end
end

