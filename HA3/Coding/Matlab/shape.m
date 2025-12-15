% Subroutine : Shape functions and its derivatives 1d
function [N,gamma] = shape(xi,nen,ndm)

% Initialisation of shape function, N, and its derivatives, gamma
N = zeros(nen,1);
gamma = zeros(nen,1);       %!!for trusses it is 1

% Shape functions and its derivatives wrt 'xi'
if nen == 2
    N(1,1) = 0.5 * (1 - xi);
    N(2,1) = 0.5 * (1 + xi);
    %
    gamma(1,1) = -0.5;
    gamma(2,1) =  0.5;
else
    error('Wrong nen..!! STOP..!!');
end

end % End of subroutine 'shape2d'
