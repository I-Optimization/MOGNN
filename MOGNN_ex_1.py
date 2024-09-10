# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 22:05:55 2024

@author: lizst
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 05:04:58 2024

@author: lizst
"""

# -*- coding: utf-8 -*-
"""
Universidade Federal Fluminense
Departamento de Engenharia Química e de Petróleo
Lizandro de Sousa Santos
E-mail: lizandrosousa@id.uff.br
"""

"""
Steepest Descent Algorithm

x0 -> initial guess
n_derive1 -> first derivative of the function
n_derive2 -> second derivative of the function (Hessian)
alpha -> optimization step (learning rate)
maxiter -> maximum number of iterations
tol -> tolerance for convergence    

"""

import numpy as np
import matplotlib.pyplot as plt
import math as m

# Objective function: calculates the objective based on P2 and P3
def func(p2, p3):
    global p1
    b = 0.286
    p4 = 10
    # Formula for the objective function
    f = ((p2 / p1)**b + (p3 / p2)**b + (p4 / p3)**b - 3)
    return f 

# First derivative (gradient) of the function
def n_derive1(x1, x2):
    # Numerical differentiation using central difference method
    grad_x1 = (func(x1 + h, x2) - func(x1 - h, x2)) / (2 * h)
    grad_x2 = (func(x1, x2 + h) - func(x1, x2 - h)) / (2 * h)
    grad = np.array([grad_x1, grad_x2])  # Gradient vector
    return grad

# Second derivative (Hessian matrix)
def n_derive2(x1, x2):
    # Numerical differentiation to compute the second derivative
    g_x0 = (n_derive1(x1 + h, x2) - n_derive1(x1 - h, x2)) / (2 * h)
    g_x1 = (n_derive1(x1, x2 + h) - n_derive1(x1, x2 - h)) / (2 * h)
    H = np.array([g_x0, g_x1])  # Hessian matrix
    return H

# Steepest descent optimization function
def steepest(x, alpha, maxiter, tol):
    x1 = x[0]
    x2 = x[1]
    erro = 1e3
    i = 0
    devx = 1e3

    # Iterative process for optimization
    while (devx > tol) or (erro > tol) or i > maxiter:
        H = n_derive2(x[0], x[1])  # Compute Hessian
        x_novo = x - alpha * np.linalg.inv(H) @ n_derive1(x[0], x[1])  # Update step
        erro = abs(func(x_novo[0], x_novo[1]) - func(x[0], x[1]))  # Error in objective function
        devx = np.linalg.norm(x_novo - x)  # Difference between successive steps

        if i > maxiter:
            break
        if x[0] > 1e5 or x[1] > 1e5:
            print('Exiting due to divergence')
            break

        i += 1
        print("Iteration", i)
        print("Error =", erro)
        print("Devx =", devx)
        x = x_novo

    return x_novo  # Return the optimized values

# Initial settings
x0 = np.array([0.2, 0.2])  # Initial guess
alpha = 1  # Learning rate
h = 1e-6  # Step size for numerical differentiation
Np = 50  # Number of points
global p1

# Optimization over multiple points
x_opt = np.zeros([Np, 2])
pp1 = np.zeros([Np])
FG = np.zeros([Np])
dist = np.ones(Np)  # Disturbance
k = 0

# Optimization loop over all disturbance values
for i in dist:
    p1 = i * (1 + np.random.random(1))[0]  # Randomize p1
    x_opt[k, :] = steepest(x0, alpha, maxiter=500, tol=1e-6)  # Perform steepest descent
    print("Optimal value of x:", x_opt[k, 0], x_opt[k, 1])
    print("Optimal function value", func(x_opt[k, 0], x_opt[k, 1]))
    pp1[k] = p1  # Store p1 value
    FG[k] = func(x_opt[k, 0], x_opt[k, 1])  # Store objective function value
    k += 1

    # Plot every 10 iterations
    if (k == 9) or (k == 19) or (k == 29) or (k == 39) or (k == 49):
        fig = plt.figure(figsize=(14, 9))
        ax = plt.axes(projection='3d')

        xa = np.linspace(1, 5, 100)
        xb = np.linspace(2, 7, 100)
        xx, yy = np.meshgrid(xa, xb)
        zz = func(xx, yy)  # Calculate function over mesh grid

        ax.set_xlabel('P1')
        ax.set_ylabel('P2')
        ax.set_zlabel('Fobj')
        ax.plot_surface(xx, yy, zz)

        # Save 3D surface and contour plots
        fig, ax = plt.subplots()
        cs = ax.contour(xx, yy, zz, levels=np.linspace(0.3, 0.8, 500))
        ax.set_xlabel('P1')
        ax.set_ylabel('P2')

        filename = str(k)
        plt.plot(x_opt[k - 1, 0], x_opt[k - 1, 1], 'o', label='Optimal point')
        plt.legend()
        plt.savefig(f'iter{filename}.png', dpi=900)
        plt.show()

# Normalization and ANN model training (for deeper analysis)
...
''
# Variables after the optimization process
X = pp1  # Store P0 (disturbance values)
pp2 = x_opt[:, 0]  # Store optimized P2
pp3 = x_opt[:, 1]  # Store optimized P3

# Normalize the values for ANN input
Xi = (X - min(X)) / (max(X) - min(X))  # Normalize X

Zia = (pp2 - min(pp2)) / (max(pp2) - min(pp2))  # Normalize P2
Zib = (pp3 - min(pp3)) / (max(pp3) - min(pp3))  # Normalize P3

# ANN Model 1 - Predicting P1 (for the optimization process)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Define and compile the first ANN model
modelA = Sequential()
modelA.add(Dense(3, input_shape=(1,), activation='relu'))  # Input layer with 3 neurons
modelA.add(Dense(50, activation='relu'))  # Hidden layer with 50 neurons
modelA.add(Dense(1, activation='sigmoid'))  # Output layer (single output)
modelA.add(Dropout(0.1))  # Dropout for regularization

modelA.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])  # Compile with loss function and optimizer

# Train the ANN model
history = modelA.fit(Xi, Zia, validation_split=0.2, epochs=150, batch_size=10)  # Train for 150 epochs

# Evaluate the model accuracy
_, accuracy = modelA.evaluate(Xi, Zia)
print('Accuracy: %.2f' % (accuracy * 100))

# Make predictions using the trained model
predictions = modelA.predict(Xi)
rounded = [round(x[0]) for x in predictions]  # Round the predictions

# Plotting the predicted and actual values of P1
T = np.linspace(1, Np, Np)
fig, ax = plt.subplots()
plt.plot(T, predictions * (max(pp2) - min(pp2)) + min(pp2), '-o', label='Optimal P1 (O-ANN model)')
plt.plot(T, Zia * (max(pp2) - min(pp2)) + min(pp2), '-*', label='Optimal P1 (theoretical model)')
ax.set_ylabel('P1')
ax.set_xlabel('Points')
plt.legend()
plt.savefig('Predição1.png', dpi=900)
plt.show()

# Plotting the disturbance (P0)
fig, ax = plt.subplots()
plt.plot(T, X, '-o', label='P0 (disturbance)')
plt.legend()
ax.set_ylabel('P0')
ax.set_xlabel('Points')
plt.savefig('Disttúrbio.png', dpi=900)
plt.show()

# ANN Model 2 - Predicting P2 (for the optimization process)
modelB = Sequential()
modelB.add(Dense(3, input_shape=(1,), activation='relu'))
modelB.add(Dense(50, activation='relu'))
modelB.add(Dense(1, activation='sigmoid'))
modelB.add(Dropout(0.1))

modelB.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse'])

# Train the second ANN model
history = modelB.fit(Xi, Zib, validation_split=0.2, epochs=150, batch_size=10)

# Evaluate model B's accuracy
_, accuracy = modelB.evaluate(Xi, Zib)
print('Accuracy: %.2f' % (accuracy * 100))

# Make predictions using the second ANN model
predictions = modelB.predict(Xi)
rounded = [round(x[0]) for x in predictions]

# Plotting the predicted and actual values of P2
fig, ax = plt.subplots()
plt.plot(T, predictions * (max(pp3) - min(pp3)) + min(pp3), '-o', label='Optimal P2 (O-ANN model)')
plt.plot(T, Zib * (max(pp3) - min(pp3)) + min(pp3), '-*', label='Optimal P2 (theoretical model)')
ax.set_ylabel('P2')
ax.set_xlabel('Points')
plt.legend()
plt.savefig('Predição2.png', dpi=900)
plt.show()

# Using the ANN model to predict P1 based on disturbances
def funcG(p1, *modelX):
    model = modelX
    X = np.array([[1], [p1]])  # Prepare the input
    f = model[0].predict(X[:, 0])  # Predict using the ANN model
    return f[1]

# Variables for storing predicted values from the ANN model
ZxA = np.zeros(Np)
ZaN = np.zeros(Np)
ZbN = np.zeros(Np)
Fobj = np.zeros(Np)
Fobj_ss1 = np.zeros(Np)
Fobj_ss2 = np.zeros(Np)
Fobj_ss3 = np.zeros(Np)
FFA = np.zeros(Np)
DA = np.linspace(0, Np, Np)
k = 0

# Loop to compute objective function for each predicted point
for i in DA:
    FFA[k] = 1 * (1 + np.random.random(1))[0]
    p1 = (FFA[k] - min(X)) / (max(X) - min(X))
    ZxA[k] = funcG(p1, modelA)
    k += 1

# Predict P2 based on disturbances using the second ANN model
ZxB = np.zeros(Np)
FFB = np.zeros(Np)
DB = np.linspace(0, Np, Np)
k = 0
for i in DB:
    p1 = (FFA[k] - min(X)) / (max(X) - min(X))
    ZxB[k] = funcG(p1, modelB)
    k += 1

# Plotting the objective function deviation based on ANN predictions
fig, ax1 = plt.subplots()
k = 0
for i in dist:
    p1 = FFA[k]
    ZaN[k] = ZxA[k] * (max(pp2) - min(pp2)) + min(pp2)
    ZbN[k] = ZxB[k] * (max(pp3) - min(pp3)) + min(pp3)
    Fobj[k] = func(ZaN[k], ZbN[k])
    Fobj_ss1[k] = func(2, 4)
    Fobj_ss2[k] = func(3, 5)
    Fobj_ss3[k] = func(4, 6)
    k += 1

# Plot of non-optimal and optimal objective function values
plt.plot(Fobj_ss1, '-', label='Non-optimal Fobj1')
plt.plot(Fobj_ss2, '-', label='Non-optimal Fobj2')
plt.plot(Fobj_ss3, '-.', label='Non-optimal Fobj3')
plt.plot(Fobj, '-o', label='Optimal Fobj (O-ANN)')
plt.legend(loc="upper left")
ax1.set_ylabel('Objective Function')
ax1.set_xlabel('Points')
plt.savefig('Fobj.png', dpi=900)
plt.show()

# Plotting the disturbance P0
fig, ax2 = plt.subplots()
plt.plot(T, FFA, '-o', label='P0 (disturbance)')
plt.legend()
ax2.set_ylabel('P0')
ax2.set_xlabel('Points')
plt.savefig('Disttúrbio_teste.png', dpi=900)
plt.show()

# Plotting the error deviations between non-optimal and optimal objective function values
fig, ax3 = plt.subplots()
plt.plot(Fobj_ss1 - Fobj, '-', label='Deviation (F1-F*)')
plt.plot(Fobj_ss2 - Fobj, '-', label='Deviation (F2-F*)')
plt.plot(Fobj_ss3 - Fobj, '-.', label='Deviation (F3-F*)')
plt.plot(np.zeros(Np), '--')
plt.legend()
ax3.set_ylabel('Deviation')
ax3.set_xlabel('Points')
plt.savefig('Erros.png', dpi=900)
plt.show()
