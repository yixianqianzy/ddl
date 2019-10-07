function nn = nnbp(nn)
%NNBP performs backpropagation
% nn = nnbp(nn) returns an neural network structure with updated weights 
    
    n = nn.n; %���������3
    sparsityError = 0;
    switch nn.output   %%**��������޸�Ϊ���ǵ�Ŀ�꺯��
        case 'sigm'
            d{n} = - nn.e .* (nn.a{n} .* (1 - nn.a{n})); %���һ���delta���в��ά�ȣ�bachsize*1
        case {'softmax','linear'}
            d{n} = - nn.e;   
        case 'tanh'
            d{n} = - nn.e .* ((1 - nn.a{n}.^2));
    end
    for i = (n - 1) : -1 : 2
        % Derivative of the activation function
        switch nn.activation_function    %ÿ��ļ����
            case 'sigm'
                d_act = nn.a{i} .* (1 - nn.a{i}); %�����ƫ��
            case 'tanh_opt'
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.a{i}.^2);
        end
        
        if(nn.nonSparsityPenalty>0)
            pi = repmat(nn.p{i}, size(nn.a{i}, 1), 1);
            sparsityError = [zeros(size(nn.a{i},1),1) nn.nonSparsityPenalty * (-nn.sparsityTarget ./ pi + (1 - nn.sparsityTarget) ./ (1 - pi))];
        end
        
        % Backpropagate first derivatives
        if i+1==n % in this case in d{n} there is not the bias term to be removed             
            d{i} = (d{i + 1} * nn.W{i} + sparsityError) .* d_act; % Bishop (5.56)  %����Ĳв�delta
        else % in this case in d{i} the bias term has to be removed
            d{i} = (d{i + 1}(:,2:end) * nn.W{i} + sparsityError) .* d_act;         %����Ĳв�delta
        end
        
        if(nn.dropoutFraction>0)
            d{i} = d{i} .* [ones(size(d{i},1),1) nn.dropOutMask{i}];
        end

    end

    for i = 1 : (n - 1)
        if i+1==n
            nn.dW{i} = (d{i + 1}' * nn.a{i}) / size(d{i + 1}, 1);   %��һ��batch�����ֵ��size��d{i+1}��=[batchsize,i+1����Ԫ����]
        else
            nn.dW{i} = (d{i + 1}(:,2:end)' * nn.a{i}) / size(d{i + 1}, 1);      
        end
    end
end
