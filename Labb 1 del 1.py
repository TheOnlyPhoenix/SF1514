#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 15:34:54 2025

@author: 
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as scp

sin = np.sin
cos = np.cos
pi = np.pi
abs = np.abs

L = 1 # balkens längd
guesses = [0.25, 0.80]
roots = []
iterations = []
diffs = []
zeros_x = []
zeros_y = []
color = ['red', 'green', 'blue']

method = 3 # 1 for FPI, 2 for NR, 3 for convergence comparison

def draw(x, f, f_x, a, b):
    x_vals = np.linspace(a, b, 1000)
    y_vals = f(x_vals, L)
    zeros_x.append(x)
    zeros_y.append(f(x))
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color=color[1])
    plt.plot(zeros_x, zeros_y, 'ro')
    ax = plt.gca()
    ax.set_ylim([-0.4, 0.4])
    plt.title(r'Balkböjning')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

def drawConvergence(iterations_1, diffs_1, iterations_2, diffs_2):    
    plt.figure(figsize=(12, 8))

    its_1 = np.linspace(0, len(iterations_1), len(diffs_1[1:]))

    plt.semilogy(its_1, diffs_1[1:], label="Fixed point iteration", color="r")

    its_2 = np.linspace(0, len(iterations_2), len(diffs_2[1:]))
    plt.semilogy(its_2, diffs_2[1:], label="Newton-Raphson", color="b")

    plt.title("Convergence")
    plt.xlabel("Iterations")
    plt.ylabel("delta(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# def drawConvergence(f, iterations, diffs, ver):
#     i = ver
      
#     plt.figure(figsize=(20, 15))
#     if i == 0:
#         its1 = np.linspace(0, iterations, 100)
#         deltas1 = np.linspace(0, iterations, 100)
    
#         plt.plot(its1, deltas1, label='Fixed point iteration', color=color[i])
#         plt.legend()
#     # elif i == 1:
#     #     its2 = np.linspace(0, iterations, 100)
#     #     deltas2 = np.linspace(0, iterations, 100)
#     #     plt.semilogy(its2, deltas2, label='Newton-Raphson', color=color[i])
#     #     plt.legend()
#     plt.title(r'Convergence')
#     plt.xlabel('Iterations')
#     plt.ylabel('delta(x)')  
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
def fpiRootFind(g, x_0, tol, max_iter):
    x = x_0
    x_i = [x_0]                
    diff_list = [tol + 1.0]

    for i in range(0, max_iter + 1):
        x_new = g(x)
        delta_x = np.abs(x_new - x) # e_i value
        if i > 0:
            x_i.append(x_new)
            diff_list.append(delta_x)
            
            if delta_x < tol:
                return x_new, i, np.array(x_i), np.array(diff_list)
        
            x = x_new
        print(i, x, g(x), delta_x)
    raise RuntimeError(
        "Fixed-point iteration did not converge within the maximum number of iterations.")
        
def newtonRootFind(f, fprim, x_0, tol, max_iter):
    x = x_0
    x_i = [x_0]
    diff_list = [tol + 1.0]
    
    for i in range(0, max_iter + 1):
        x_new = x - f(x)/fprim(x)
        delta_x = np.abs(x_new-x)
        if i > 0:
            x_i.append(x_new)
            diff_list.append(delta_x)
            if delta_x < tol:
                return x_new, i, np.array(x_i), np.array(diff_list)
            x = x_new 
        print(i, x, f(x), delta_x)
    raise RuntimeError(
        "Fixed-point iteration did not converge within the maximum number of iterations.")

def pickMethod(case, f, g, tol, max_iter, fprim):
    match case:
        case 1:
            root, iteration, x_i, diffs = fpiRootFind(g, guesses[1], tol, max_iter)
            draw(root, f, f(root), -0.5, 1.5)
            print('\n')
            print(f"FPI: Root: {root}, after {iteration} iterations")
        case 2:
            for i in range(0, 2):
                print('\n')
                root, iteration, x_i, diffs = newtonRootFind(f, fprim, guesses[i], tol, max_iter)
                roots.append(root)
                iterations.append(iteration)
                draw(roots[i], f, f(roots[i]), -0.5, 1.5)
                print('\n')
                print(f"NR: Root {i+1}: {roots[i]}, after {iterations[i]} iterations")
        case 3:
            root_FPI, iteration_FPI, iterations_1, diffs_1 = fpiRootFind(g, guesses[1], tol, max_iter)
            print('\n')
            print(f"FPI: Root: {root_FPI}, after {iteration_FPI} iterations")

            print('\n')
            root_NR, iteration_NR, iterations_2, diffs_2 = newtonRootFind(f, fprim, guesses[1], tol, max_iter)
            print(f"NR: Root: {root_NR}, after {iteration_NR} iterations")
            
            drawConvergence(iterations_1, diffs_1, iterations_2, diffs_2)

def main():
    tol = 10**-10
    max_iter = 1000
    f = lambda x, L = 1: 8/3*(x/L) - 3*(x/L)**2 + 1/3*(x/L)**3 - 2/3*sin(pi*x/L)
    g = lambda x, L = 1: 3*L/8*(3*(x/L)**2-1/3*(x/L)**3+2/3*sin(pi*x/L))
    fprim = lambda x, L = 1: 8/3*L - 6*(x/L) + (x/L)**2 - pi/L*2/3*cos(pi*x/L)
    pickMethod(method, f, g, tol, max_iter, fprim)
    

    
if __name__=="__main__":
    main()
    
# 1)
# a) 2 roots according to graph
# b) leftmost root (near x=0.25) has a convergence value >1, which means it cannot be calculated using fixed point iteration
# rightmost root has a convergence value <1, so it can be calculated using FPI.
