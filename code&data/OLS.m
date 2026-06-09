function [beta, L] = OLS(X,Y)
beta = (X'*X)^-1*X'*Y;
e = Y-X*beta;
L = e'*e;
end