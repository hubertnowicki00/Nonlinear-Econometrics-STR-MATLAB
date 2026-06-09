function L = loss(y,x,s,param)
    % [beta1,beta2,lam,c]

    K = size(x,2); % no of columns
    beta1 = param(1:K);
    beta2 = param(K+1:2*K);
    lam = param(2*K+1);
    c = param(2*K+2);
    
    g = 1./(1+ exp(-lam*(s-c)));
    model = x*beta1' + g.*x*beta2';
    e = y - model;
    L = sum(e.^2);
end