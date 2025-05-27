#problem12_euler_methods.py
import matplotlib.pyplot as plt
import numpy as np

def dydx(x, y):
    return -y + x + 1

def euler(f, x0, y0, h, n):
    xs = [x0]
    ys = [y0]
    for i in range(n):
        y0 = y0 + h * f(x0, y0)
        x0 = x0 + h
        xs.append(x0)
        ys.append(y0)
    return xs, ys

def improved_euler(f, x0, y0, h, n):
    xs = [x0]
    ys = [y0]
    for i in range(n):
        y_pred = y0 + h * f(x0, y0)
        y0 = y0 + h/2 * (f(x0, y0) + f(x0 + h, y_pred))
        x0 = x0 + h
        xs.append(x0)
        ys.append(y0)
    return xs, ys

def analytic_solution(x):
    return x + np.exp(1 - x)

# Parameters
x0, y0, h, n = 1.0, 2.0, 0.2, 20
x_vals = np.linspace(x0, x0 + h*n, 200)
y_exact = analytic_solution(x_vals)

# Run methods
x_euler, y_euler = euler(dydx, x0, y0, h, n)
x_improved, y_improved = improved_euler(dydx, x0, y0, h, n)

# Plotting
plt.plot(x_vals, y_exact, 'k-', label='Analytic')
plt.plot(x_euler, y_euler, 'ro--', label='Euler')
plt.plot(x_improved, y_improved, 'bs--', label='Improved Euler')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Euler vs Improved Euler')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("problem12_euler_methods.png")
