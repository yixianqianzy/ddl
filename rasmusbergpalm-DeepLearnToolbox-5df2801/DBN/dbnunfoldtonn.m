function nn = dbnunfoldtonn(dbn, outputsize)
%DBNUNFOLDTONN Unfolds a DBN to a NN
%   dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
%   layer of size outputsize added. 目标输出的label个数；比如在MINST就是10，DBN只负责学习feature  
%   或者说初始化Weight，是一个unsupervised learning，最后的supervised还得靠NN  
    if(exist('outputsize','var'))    %检测“outputsize” 类型为“变量”是否存在
        size = [dbn.sizes outputsize];   %784*100*10
    else
        size = [dbn.sizes];  
    end
    nn = nnsetup(size);
    for i = 1 : numel(dbn.rbm)
        nn.W{i} = [dbn.rbm{i}.c dbn.rbm{i}.W]; %将DBN预训练得到的每一层权重+偏置——NN的每层权重初始化
    end
end

