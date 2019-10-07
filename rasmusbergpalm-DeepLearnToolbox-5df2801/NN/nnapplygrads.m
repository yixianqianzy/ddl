function nn = nnapplygrads(nn)
%NNAPPLYGRADS updates weights and biases with calculated gradients
% nn = nnapplygrads(nn) returns an neural network structure with updated
% weights and biases
    
    for i = 1 : (nn.n - 1)
        if(nn.weightPenaltyL2>0) %如果有权重的L2正则化项（惩罚项)
            dW = nn.dW{i} + nn.weightPenaltyL2 * [zeros(size(nn.W{i},1),1) nn.W{i}(:,2:end)]; %w的增量
        else
            dW = nn.dW{i};
        end
        
        dW = nn.learningRate * dW; %增量*学习率
        
        if(nn.momentum>0) 
            nn.vW{i} = nn.momentum*nn.vW{i} + dW; 
            dW = nn.vW{i};
        end
            
        nn.W{i} = nn.W{i} - dW;
    end
end
