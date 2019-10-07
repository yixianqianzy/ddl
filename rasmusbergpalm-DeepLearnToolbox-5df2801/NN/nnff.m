function nn = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n; %层数：3层
    m = size(x, 1); % 数据输入维度：784
    
    x = [ones(m,1) x]; %将第一列数据当作1,增加偏置的输入神经元
    nn.a{1} = x;  % 第一层的输出

    %feedforward pass
    for i = 2 : n-1  
        switch nn.activation_function    %计算激活函数的值: %%** 激活函数可以是sigm或者softmax
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
     %   m=nn.size(i);
        nn.a{i} = [ones(m,1) nn.a{i}];  %添加偏置项对应的输入：方便下一次的迭代计算
    end
    
    switch nn.output    %%**这里可以修改为我们的目标函数
        case 'sigm'
            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
        case 'tanh'
            nn.a{n}=tanh(nn.a{n - 1} * nn.W{n - 1}');
    end

    %error and loss
    nn.e = y - nn.a{n};   %前向误差（真实目标y - 前向计算出的输出）；维度：batchsize的个数，
    
    switch nn.output   %%**这里可以修改为我们的目标函数
        case {'sigm', 'linear', 'tanh'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m;  %对所有神经元的平方损失的平均值（对m个样本）
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m; %最后一层输出是softmax函数时的损失 %%**这里可以修改为我们的目标函数
    end
end
