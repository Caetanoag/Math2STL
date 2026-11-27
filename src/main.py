from src.parser import parse_equation
from src.plot_exp2d import plot_expression_gpu

nome, variaveis, expr, raw_str, f_num = parse_equation("f(x, y) = sqrt(x**2 + y**2)")

plot_expression_gpu(
    expr=expr,
    vars_list=variaveis,
    x_range=(-100, 100),
    y_range=(-100, 100),
    num_points=150,
)
