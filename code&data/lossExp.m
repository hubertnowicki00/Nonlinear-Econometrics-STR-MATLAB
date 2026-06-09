function L = lossExp(y,x,s,param)
    k = size(x,2);
    beta1 = param(1:k);
    beta2 = param(k+1:2*k);
    lambd = param(end-1);
    c = param(end);
    g = 1 - exp(-lambd*(s-c).^2);
    
    if isrow(g)
        g = g';
    end

    err = y - (x*beta1+(g.*x)*beta2); 
    L = sum(err.^2);
end