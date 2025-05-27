#problem2_root_finding.py
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**3 - 2*x**2 + 3*x - 5

def df(x):
    return 3*x**2 - 4*x + 3

def newton_method(f, df, x0, tol=1e-6, max_iter=100):
    x = x0
    history = [x]
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-10:
            break
        x_new = x - fx / dfx
        history.append(x_new)
        if abs(x_new - x) < tol:
            break
        x = x_new
    return history

def secant_method(f, x0, x1, tol=1e-6, max_iter=100):
    history = [x0, x1]
    for _ in range(max_iter):
        fx0, fx1 = f(x0), f(x1)
        if abs(fx1 - fx0) < 1e-10:
            break
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        history.append(x2)
        if abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return history

def bisection_method(f, a, b, tol=1e-6, max_iter=100):
    history = []
    for _ in range(max_iter):
        c = (a + b) / 2
        history.append(c)
        if abs(f(c)) < tol or abs(b - a) < tol:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return history

# Visualization
x_vals = np.linspace(0, 3, 400)
y_vals = f(x_vals)
newton_hist = newton_method(f, df, 3.0)
secant_hist = secant_method(f, 2.5, 3.0)
bisection_hist = bisection_method(f, 1.0, 3.0)

plt.plot(x_vals, y_vals, label='f(x)', color='black')
plt.axhline(0, color='gray', linestyle='--')
plt.plot(newton_hist, [f(x) for x in newton_hist], 'o-', label='Newton')
plt.plot(secant_hist, [f(x) for x in secant_hist], 's-', label='Secant')
plt.plot(bisection_hist, [f(x) for x in bisection_hist], 'x-', label='Bisection')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Root Finding Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("problem2_root_methods.png")
