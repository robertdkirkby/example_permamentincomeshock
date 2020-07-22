function F=MultiplicativePermanentIncomeShocks_ReturnFn(aprime,a,psi,theta,R,g,gamma)

% Because of how the value function is defined P is normalized to one

c=R*a/(g*psi)+theta-aprime;

F=-Inf;
if c>0
    F=(c^(1-gamma))/(1-gamma);
end

end