function sae = saetrain(sae, x, opts)
% x=train_x;
    for i = 1 : numel(sae.ae);
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
      nn = sae.ae{i};
      %nn.activation_function              = 'sigm';   %  Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
%       nn.momentum                         = 0.9;          %  Momentum
%       nn.scaling_learningRate             = 0.9;            %  Scaling factor for the learning rate (each epoch)
%       nn.weightPenaltyL2                  = 0 ;            %  L2 regularization
%       nn.nonSparsityPenalty               = 0;            %  Non sparsity penalty
%       nn.sparsityTarget                   = 0.05;         %  Sparsity target
%       nn.inputZeroMaskedFraction          = 0.1;            %  Used for Denoising AutoEncoders
%       nn.dropoutFraction                  = 0;            %  Dropout level (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
%       nn.testing                          = 0;            %  Internal variable. nntest sets this to one.
      nn.output                           = 'sigm';       %  output unit 'sigm' (=logistic), 'softmax' and 'linear'
      
        sae.ae{i} = nntrain(nn, x, x, opts);
        t = nnff(sae.ae{i}, x, x);
        x = t.a{2};
        %remove bias term
        x = x(:,2:end);
    end
end
