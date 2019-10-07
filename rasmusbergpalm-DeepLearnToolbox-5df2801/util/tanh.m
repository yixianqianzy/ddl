function X = tanh(P)
    X = (exp(P)-exp(-P))./(exp(P)+exp(-P));
end