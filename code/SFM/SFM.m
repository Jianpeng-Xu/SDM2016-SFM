function [W_U, W_I] = SFM(R, X_U, X_I, lambda1, lambda2, k)

% n: number of users
% m: number of items
% R: n x m
% X_I: m x d
% X_U: n x d
% W_I: k x d -- Q in paper
% W_U: d x k -- P in paper

[m,d] = size(X_I);

W_I = rand(k,d);
W_U = rand(d, k);
Wvect0 = [W_I(:); W_U(:)];

% function value/gradient of the smooth part
smoothF    = @(parameterVect) smooth_part( parameterVect, R, X_I, X_U, d, k);
% non-negativen l1 norm proximal operator.
non_smooth = prox_P(lambda1, lambda2,d,k);

sparsa_options = pnopt_optimset(...
    'display'   , 1    ,...
    'debug'     , 0    ,...
    'maxIter'   , 200  ,...
    'ftol'      , 1e-5 ,...
    'optim_tol' , 1e-5 ,...
    'xtol'      , 1e-5 ...
    );
[W_vect, ~,info] = pnopt_sparsa( smoothF, non_smooth, Wvect0, sparsa_options );

W_I = reshape (W_vect(1:d*k), [k, d]);
W_U = reshape (W_vect(d*k+1:end), [d,k]);

end

function [f, g] = smooth_part(parameterVect, R, X_I, X_U, d, k)
W_I = reshape (parameterVect(1:d*k), [k, d]);
W_U = reshape (parameterVect(d*k+1:end), [d,k]);

% compute f
XUWU = X_U * W_U;
WIXI = W_I * X_I';
XWWX = XUWU * WIXI;
f = norm(R - XWWX, 'fro')^2;

% compute g_W_I
g_W_I = 2*XUWU' * (XWWX - R) * X_I;
% compute g_W_U
g_W_U = 2*(WIXI * (XWWX - R)' * X_U)';
g = [g_W_I(:); g_W_U(:)];

end


function op = prox_P( lambda1 , lambda2,d,k) %lambda1, lambda2, lambdaR, k,d

%PROX_L1    L1 norm.
%    OP = PROX_L1( q ) implements the nonsmooth function
%        OP(X) = norm(q.*X,1).
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
% Dual: proj_linf.m

% Update Feb 2011, allowing q to be a vector
% Update Mar 2012, allow stepsize to be a vector

if nargin == 0
    lambda1 = 1;
elseif ~isnumeric( lambda1 ) || ~isreal( lambda1 ) ||  any( lambda1 < 0 ) %|| all(lambda1==0) %|| numel( q ) ~= 1
    error( 'Argument must be positive.' );
end

op = tfocs_prox( @f, @prox_f , 'vector' ); % Allow vector stepsizes
    function v = f(x)
        W_I = reshape (x(1:d*k), [k, d]);
        W_U = reshape (x(d*k+1:end), [d,k]);
        v = lambda1 * sum(sqrt(sum(W_I.^2))) + lambda2 * sum(sqrt(sum(W_U.^2, 2)));
    end

    function x = prox_f(x,t)
        t1 = t .* lambda1; % March 2012, allowing vectorized stepsizes
        t2 = t .* lambda2;
        W_I = reshape (x(1:d*k), [k, d]);
        W_U = reshape (x(d*k+1:end), [d,k]);
        
        % project W: Group Lasso norm
        W_I= repmat(max(0, 1 - t1./sqrt(sum(W_I.^2, 1))), [k, 1]) .* W_I;
        % project U: Group Lasso norm
        W_U = repmat(max(0, 1 - t1./sqrt(sum(W_U.^2, 2))), [1, k]) .* W_U;
        
        % put W and U back to x
        x(1:d*k) = W_I(:);
        x(d*k+1:end) = W_U(:);
    end

end
