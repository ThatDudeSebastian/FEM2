% Subroutine : Gauss quadrature 1d
function [xi,w8] = gauss(nqp,ndm)

if nqp == 1
    xi = 0;
    w8 = 2;
else
    error('Wrong nqp..!! STOP..!!');
end
    
end % End of subroutine 'gauss2d'
