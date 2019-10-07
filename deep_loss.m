function L = deep_loss(D, nn)
dif_mat = D - nn.a{nn.n}';
L = norm(dif_mat,'fro')^2;
end
