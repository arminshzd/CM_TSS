clc;clear;

V = @(x, y) (1-y^2)*x^2*exp(-x^2)+1/2*y^2;
dVdx = @(x, y) 2*x*exp(-x^2)*(1-x^2)*(1-y^2);
dVdy = @(x, y) y*(1-2*x^2*exp(-x^2));
d2Vdx2 = @(x, y) (4*x^4-10*x^2+2)*(1-y^2)*exp(-x^2);
d2Vdydx = @(x, y) (2*x*exp(-x^2)-2*x^3*exp(-x^2))*(-2*y);
d2Vdy2 = @(x, y) (1-2*x^2*exp(-x^2));

a = [0.0001; 0.0001];
Flag = true;
Flag_in = false;
cnt1 = 0;
cnt2 = 0;
while Flag
    cnt1 = cnt1 + 1;
    V0 = V(a(1), a(2));
    D = [dVdx(a(1), a(2)); dVdy(a(1), a(2))];
    K = [d2Vdx2(a(1), a(2)), d2Vdydx(a(1), a(2)); d2Vdydx(a(1), a(2)), d2Vdy2(a(1), a(2))];
    [Up, em, U] = eig(K);
    eigv = diag(em);
    d = Up*D;
    F = @(l) d(1)^2./(l-eigv(1)).^3 + d(2)^2./(l-eigv(2)).^3;
    xs = linspace(-100,100,2001);
    ys = F(xs);
    scinter = find(diff(sign(ys)));
    ninter = numel(scinter);
    xroots = NaN(1,ninter);
    for i = 1:ninter
        xroots(i) = fzero(F,xs(scinter(i) + [0 1]));
    end
    lambda = min(xroots);
    if and(lambda < 0, abs(lambda)>1e-3)
        Flag_in = true;
        lambda = 0;
        cnt2 = cnt2 + 1;
    end
    dx = inv(lambda.*ones(2,2)-K)*D;
    aold = a;
    a = aold + dx;
    if Flag_in
        if and(abs(aold(1)-a(1))<1e-6, abs(aold(2)-a(2))<1e-6)
            Flag = false;
        end
    end
end
