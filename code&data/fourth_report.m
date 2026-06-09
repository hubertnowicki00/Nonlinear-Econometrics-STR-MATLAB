%% task 4.1
% simulate smooth transition model and compute the loss score
t = linspace(-2*pi, 2*pi, 500);
s = cos(t);
lambd = 3;
c = 0.3;
g = 1-exp(-lambd*(s-c).^2);

X = [cos(t/4)', ones(length(t), 1)];
b1 = [1; 2];
b2 = [2; 1];

e = normrnd(0, 1, size(g'));
y = X*b1+(g'.*X)*b2+e;
figure;
plot(t, y, 'DisplayName', 'y(t)');
hold on;
plot(t, g, 'DisplayName', 'G(t)', 'LineWidth', 1.5);
title('Exp STR model');
legend;

param = [b1; b2; lambd; c];
L = lossExp(y, X, s, param);

disp('Loss L is below:')
L % 470.9648


%% task 4.2
clear variables
clc
load('Report4_2.mat');

k = size(x, 2);
lambdas = logspace(-2, 2, 50);
cs = linspace(quantile(s, 0.1), quantile(s, 0.9), 25);
losses = nan(length(lambdas), length(cs));

for i = 1:length(lambdas) % grid search
    for j = 1:length(cs)
        
        lam = lambdas(i);
        c = cs(j);
        G = 1./(1+exp(-lam*(s-c))); 
        Z2 = (1-G).*x;
        Z  = G.*x;
        
        X2 = [Z2, Z];
        [betas_ols, ~] = OLS(X2, y);
        b1_ols = betas_ols(1:k);
        b2_star_ols = betas_ols(k+1:end);
        
        b2_ols = b2_star_ols - b1_ols;
        param_vector = [b1_ols', b2_ols', lam, c];
        losses(i, j) = loss(y, x, s, param_vector);
        
    end
end

[min_loss, linear_idx] = min(losses(:));
[opt_lam_idx, opt_c_idx] = ind2sub(size(losses), linear_idx);

lam_opt = lambdas(opt_lam_idx);
c_opt = cs(opt_c_idx);
G_opt = 1./(1+exp(-lam_opt*(s-c_opt)));
Z2_opt = (1-G_opt).*x;
Z_opt = G_opt.*x;
X2_opt = [Z2_opt, Z_opt];

[betas_opt, ~] = OLS(X2_opt, y);
b1_opt = betas_opt(1:k);
b2_star_opt = betas_opt(k+1:end);
b2_opt = b2_star_opt - b1_opt;

disp('Minimal loss score: ')
min_loss
disp('Optimal lambda val: ')
lam_opt
disp('Optimal c val: ')
c_opt
disp('Optimal b1 val: ')
b1_opt
disp('Optimal b2 val: ')
b2_opt

figure;
heatmap(cs, lambdas, losses, 'Colormap', jet);
title('Grid search loss');
xlabel('c val');
ylabel('lambda'); % heatmap

%% task 4.3
clear variables
clc
load('Report4_3.mat'); 
params0 = [beta1; beta2; lam; c]'; 
fun = @(params) loss(y, x, s, params);
loss_initial = fun(params0);

disp('fminunc')
opts_unc = optimoptions('fminunc', 'MaxFunctionEvaluations', 5000, 'MaxIterations', 2000);
optim_params_unc = fminunc(fun, params0, opts_unc);
loss_unc = fun(optim_params_unc);

disp('fmincon')
K = size(x, 2);
lb = [-inf(1, 2*K), 0.01, quantile(s, 0.1)];
up = [inf(1, 2*K), 100, quantile(s, 0.9)];

optim_params_con = fmincon(fun, params0, [], [], [], [], lb, up);
loss_con = fun(optim_params_con);

disp('fmincon & step tolerance of 0.05')
opts_con = optimoptions('fmincon', 'StepTolerance', 0.05);
optim_params_con_tol = fmincon(fun, params0, [], [], [], [], lb, up, [], opts_con);
loss_con_tol = fun(optim_params_con_tol);

% table with comparisons
Methods = {'Initial'; 'fminunc'; 'fmincon'; 'fmincon_StepTol'};
Params_Initial = params0;
Params_Unc = optim_params_unc;
Params_Con = optim_params_con;
Params_Con_Tol = optim_params_con_tol;
All_Params = [Params_Initial; Params_Unc; Params_Con; Params_Con_Tol];

Lambda_Est = [params0(end-1); optim_params_unc(end-1); optim_params_con(end-1); optim_params_con_tol(end-1)];
C_Est = [params0(end); optim_params_unc(end); optim_params_con(end); optim_params_con_tol(end)];
Loss_Scores = [loss_initial; loss_unc; loss_con; loss_con_tol];

ResultsTable = table(Methods, All_Params, Lambda_Est, C_Est, Loss_Scores);
disp('Comparison table')
disp(ResultsTable);



%% ADDITIONAL TASK 4.4 - PART 1 - DATA PREPARATION

countrycode = 'POL'; 
rawData = readtable('data_combined.xlsx', 'VariableNamingRule', 'preserve');
country_rows = strcmp(string(rawData{:, 2}), countrycode);
POLSKAGUROM = rawData(country_rows, :);

if isempty(POLSKAGUROM)
    error(['Country code ', countrycode, ' not found in the dataset.']);
end

headers = POLSKAGUROM.Properties.VariableNames;
year_headers = headers(5:end);
Years = zeros(length(year_headers), 1);

for i = 1:length(year_headers)
    Years(i) = str2double(year_headers{i}(1:4));
end

series_names = string(POLSKAGUROM{:, 3});
gdp_idx = contains(series_names, 'GDP growth');
unemp_idx = contains(series_names, 'Unemployment');
inf_idx = contains(series_names, 'Inflation');
res_idx = contains(series_names, 'Electricity production');

GDP = str2double(string(table2cell(POLSKAGUROM(gdp_idx, 5:end))))';
Unemp = str2double(string(table2cell(POLSKAGUROM(unemp_idx, 5:end))))';
Inf = str2double(string(table2cell(POLSKAGUROM(inf_idx, 5:end))))';
RES = str2double(string(table2cell(POLSKAGUROM(res_idx, 5:end))))';

% t-1 lags
y = GDP(2:end);
Years_model = Years(2:end); 
X = [Unemp(1:end-1), Inf(1:end-1), RES(1:end-1)];
s = Unemp(1:end-1); % transition variable is Unemployment t-1 !!!

missing_rows = isnan(y) | any(isnan(X), 2); % NaN handling
y(missing_rows) = [];
X(missing_rows, :) = [];
s(missing_rows) = [];
Years_model(missing_rows) = []; 
n = length(y); 

disp(['Data loaded for ', countrycode, '. N observations = ', num2str(n)]);

% quick dataset check using plots because it is more convenient
% than looking at matrixes

figure('Name', ['Dataset overview for ', countrycode], 'Position', [100, 100, 800, 600]);

% GDP subplot
subplot(2, 2, 1);
plot(Years_model, y, '-o', 'Color', [0 0.4470 0.7410], 'LineWidth', 1.5);
title('Annual GDP');
ylabel('%');
grid on;

% unemployment subplot
subplot(2, 2, 2);
plot(Years_model, X(:, 1), '-s', 'Color', [0.8500 0.3250 0.0980], 'LineWidth', 1.5);
title('Unemployment');
ylabel('% of total labor force');
grid on;

% inflation subplot
subplot(2, 2, 3);
plot(Years_model, X(:, 2), '-^', 'Color', [0.9290 0.6940 0.1250], 'LineWidth', 1.5);
title('Inflation');
ylabel('%');
grid on;

% RES subplot
subplot(2, 2, 4);
plot(Years_model, X(:, 3), '-d', 'Color', [0.4660 0.6740 0.1880], 'LineWidth', 1.5);
title('Renewable energy');
ylabel('% of total energy');
xlabel('Year');
grid on;

sgtitle(['Dataset overview for ', countrycode]);


%% ADDITIONAL TASK 4.4 - PART 2 - MODEL COMPARISON

% STR logisitc model
LAMBDA_log = logspace(0.01, 100, 50);
C_grid = linspace(quantile(s, 0.1), quantile(s, 0.9), 25);
K = size(X, 2);

LS_log = zeros(length(LAMBDA_log), length(C_grid)); % grid search
for i = 1:length(LAMBDA_log)
    for j = 1:length(C_grid)
        lam = LAMBDA_log(i);
        c = C_grid(j);
        g = 1 ./ (1 + exp(-lam * (s - c)));
        X2 = [X, g .* X];
        [~, LS_log(i,j)] = OLS(X2, y); 
    end
end

[ii, jj] = find(LS_log == min(min(LS_log)));
lam_init = LAMBDA_log(ii(1));
c_init = C_grid(jj(1));
g = 1 ./ (1 + exp(-lam_init * (s - c_init)));
X2 = [X, g .* X];

[beta_init, ~] = OLS(X2, y);
params0_log = [beta_init', lam_init, c_init];

% fmincon for STR logistic
fun_log = @(params) loss(y, X, s, params);
lb = [-inf(1, 2*K), 0.01, quantile(s, 0.1)];
up = [inf(1, 2*K), 100, quantile(s, 0.9)];

opts = optimoptions('fmincon', 'Display', 'off');
optim_log = fmincon(fun_log, params0_log, [], [], [], [], lb, up, [], opts);

beta1_log = optim_log(1:K)'; % predictions
beta2_log = optim_log(K+1:2*K)';
lam_log = optim_log(end-1);
c_log = optim_log(end);
G_log = 1 ./ (1 + exp(-lam_log * (s - c_log)));
y_hat_log = X*beta1_log + G_log .* X * beta2_log;
RSS_log = sum((y - y_hat_log).^2);


% STR exponential model
LAMBDA_exp = logspace(0.01, 100, 50);

LS_exp = zeros(length(LAMBDA_exp), length(C_grid)); % grid search
for i = 1:length(LAMBDA_exp)
    for j = 1:length(C_grid)
        lam = LAMBDA_exp(i);
        c = C_grid(j);
        g = 1 - exp(-lam * (s - c).^2);
        X2 = [X, g .* X];
        [~, LS_exp(i,j)] = OLS(X2, y); 
    end
end

[ii, jj] = find(LS_exp == min(min(LS_exp)));
lam_init_exp = LAMBDA_exp(ii(1));
c_init_exp = C_grid(jj(1));
g = 1 - exp(-lam_init_exp * (s - c_init_exp).^2);
X2 = [X, g .* X];

[beta_init_exp, ~] = OLS(X2, y);
params0_exp = [beta_init_exp; lam_init_exp; c_init_exp];
lb_exp = lb';
up_exp = up';

% fmincon for STR exponential
fun_exp = @(params) lossExp(y, X, s, params);
optim_exp = fmincon(fun_exp, params0_exp, [], [], [], [], lb_exp, up_exp, [], opts);

beta1_exp = optim_exp(1:K); % predictions
beta2_exp = optim_exp(K+1:2*K);
lam_exp = optim_exp(end-1);
c_exp = optim_exp(end);
G_exp = 1 - exp(-lam_exp * (s - c_exp).^2);
y_hat_exp = X*beta1_exp + G_exp .* X * beta2_exp;
RSS_exp = sum((y - y_hat_exp).^2);


% Linear regression model - using OLS for simplicity
[beta_lin, RSS_lin] = OLS(X, y); 
y_hat_lin = X * beta_lin;


% plotting
figure;
plot(Years_model, y, 'w-', 'LineWidth', 1); hold on;
plot(Years_model, y_hat_log, 'b--', 'LineWidth', 1.5);
plot(Years_model, y_hat_exp, 'r:', 'LineWidth', 1.5);
plot(Years_model, y_hat_lin, 'g-.', 'LineWidth', 1.5);
hold off;

title(['GDP modeling for ', countrycode]);
xlabel('Year');
ylabel('GDP growth');
legend('Real GDP', 'Logistic STR', 'Exponential STR', 'Linear Regression', 'Location', 'southwest');
grid on;


% BIC comparison
k_lin = 3;
k_str = 8; 

BIC_log = n * log(RSS_log / n) + k_str * log(n);
BIC_exp = n * log(RSS_exp / n) + k_str * log(n);
BIC_lin = n * log(RSS_lin / n) + k_lin * log(n);

Models = {'Logistic STR'; 'Exponential STR'; 'Linear Regression'};
BIC_Scores = [BIC_log; BIC_exp; BIC_lin];
RSS_Scores = [RSS_log; RSS_exp; RSS_lin];

ComparisonTable = table(Models, RSS_Scores, BIC_Scores);
disp('Model comparison');
disp(ComparisonTable);