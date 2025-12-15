% Subroutine : Jacobian 'matrix'
function [detJq,invJq] = jacobian(xe,gamma,nen,ndm)

% Jacobian matrix at the quadrature points q
% Hint : For 1d-case the Jacobian is a scalar
% Jq = xe * gamma;
%
Jq = xe*gamma;
        
% Determinant of Jacobian
detJq = det(Jq);

% Inverse of Jacobian
invJq = inv(Jq);

end % End of subroutine 'jacobian2d'
