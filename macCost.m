function alm = macCost(K, w_n )

coef_mul = [269.9581 ; -1.4028];
coef_sum = [  7.1791 ; -0.0623];
coef_sum(1) = coef_sum(1) ./ 121 .* K .* K

w_sum = w_n(:,1);
w_mul = w_n(:,2);

x_sum = [ones(length(w_sum),1) w_sum];
x_mul = [ones(length(w_mul),1) w_mul];

alm = x_sum*coef_sum + x_mul*coef_mul ;
end

