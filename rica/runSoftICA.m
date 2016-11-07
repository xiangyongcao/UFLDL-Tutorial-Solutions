%% We will use minFunc for this exercise, but you can use your
% own optimizer of choice
clear all;
addpath(genpath('../common/')) % path to minfunc
%% These parameters should give you sane results. We recommend experimenting
% with these values after you have a working solution.
global params;
params.m=10000; % num patches
params.patchWidth=9; % width of a patch
params.n=params.patchWidth^2; % dimensionality of input to RICA
params.lambda = 0.0005; % sparsity cost
params.numFeatures = 50; % number of filter banks to learn
params.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity

% Load MNIST data set
data = loadMNISTImages('../common/train-images-idx3-ubyte');

%% Preprocessing
% Our strategy is as follows:
% 1) Sample random patches in the images
% 2) Apply standard ZCA transformation to the data
% 3) Normalize each patch to be between 0 and 1 with l2 normalization

% Step 1) Sample patches
patches = samplePatches(data,params.patchWidth,params.m);
% Step 2) Apply ZCA
patches = zca2(patches);
% Step 3) Normalize each patch. Each patch should be normalized as
% x / ||x||_2 where x is the vector representation of the patch
m = sqrt(sum(patches.^2) + (1e-8));
x = bsxfunwrap(@rdivide,patches,m);

%% Run the optimization
options.Method = 'lbfgs';
options.MaxFunEvals = Inf;
options.MaxIter = 500;
%options.display = 'off';
options.outputFcn = @showBases;

% initialize with random weights
randTheta = randn(params.numFeatures,params.n)*0.01; % 1/sqrt(params.n);
randTheta = randTheta ./ repmat(sqrt(sum(randTheta.^2,2)), 1, size(randTheta,2));
randTheta = randTheta(:);

%% Gradient Checking
DEBUG = 1;  % set this to true to check gradient
if DEBUG
    
    paramsGC.m=200; % num patches
    paramsGC.patchWidth=4; % width of a patch
    paramsGC.n=paramsGC.patchWidth^2; % dimensionality of input to RICA
    paramsGC.lambda = 0.0005; % sparsity cost
    paramsGC.numFeatures = 10; % number of filter banks to learn
    paramsGC.epsilon = 1e-2; % epsilon to use in square-sqrt nonlinearity
    
    patches = samplePatches(data,paramsGC.patchWidth,paramsGC.m);
    patches = zca2(patches);
    m = sqrt(sum(patches.^2) + (1e-8));
    xGC = bsxfunwrap(@rdivide,patches,m);
    
    % initialize with random weights
    Theta0 = randn(paramsGC.numFeatures,paramsGC.n)*0.01; % 1/sqrt(params.n);
    Theta0 = Theta0 ./ repmat(sqrt(sum(Theta0.^2,2)), 1, size(Theta0,2));
    Theta0 = Theta0(:);
    
    [cost,grad] = softICACost(Theta0, xGC, paramsGC);
    
    % Check gradients
    numGrad = computeNumericalGradient( @(theta) softICACost(theta, xGC, paramsGC), Theta0);
    
    % Use this to visually compare the gradients side by side
    disp([numGrad grad]);
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    % Should be small. In our implementation, these values are usually
    % less than 1e-9.
    disp(diff);
    
    assert(diff < 1e-6,...
        'Difference too large. Check your gradient computation again');
    
end;

% optimize
[opttheta, cost, exitflag] = minFunc( @(theta) softICACost(theta, x, params), randTheta, options); % Use x or xw

% display result
W = reshape(opttheta, params.numFeatures, params.n);
display_network(W');
