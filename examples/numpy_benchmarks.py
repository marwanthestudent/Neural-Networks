import numpy as np
import time


def matrix_addition(first, second):
    (rows1, cols1) = first.shape
    result = np.zeros((rows1, cols1))

    for i in range(0, rows1):
        for j in range(0, cols1):
            result[i,j] = first[i,j] + second[i,j]
    
    return result


def matrix_multiplication(first, second):
    (rows1, cols1) = first.shape
    (rows2, cols2) = second.shape
    result = np.zeros((rows1, cols2))

    for i in range(0, rows1):
        for j in range(0, cols2):
            sum = 0
            for k in range(0, rows2):
                sum = sum + first[i,k]*second[k,j]
            result[i,j] = sum
    
    return result

#%%

rows = 5000
cols = rows
A = np.random.rand(rows, cols)
B = np.random.rand(rows, cols)

start_time = time.time_ns()
result = matrix_addition(A, B)
end_time = time.time_ns()
#print(result)

total_time = (end_time-start_time) / 1000000000
print("My addition, size %dx%d, total time: %.5f seconds" % 
      (rows, cols, total_time))


start_time = time.time_ns()
result = A + B
end_time = time.time_ns()
#print(result)

total_time = (end_time-start_time) / 1000000000
print("np addition, size %dx%d, total time: %.5f seconds" % 
      (rows, cols, total_time))

"""
Output on my computer: 
My addition, size 5000x5000, total time: 9.68999 seconds
np addition, size 5000x5000, total time: 0.05838 seconds
My function was about 166 times slower.

Output on Google Colab: 
My addition, size 5000x5000, total time: 18.58570 seconds
np addition, size 5000x5000, total time: 0.14109 secondsMy function was about 166 times slower.
My function was about 131 times slower.
"""

#%%

rows = 500
cols = rows
A = np.random.rand(rows, cols)
B = np.random.rand(rows, cols)

start_time = time.time_ns()
result = matrix_multiplication(A, B)
end_time = time.time_ns()
#print(result)

total_time = (end_time-start_time) / 1000000000
print("My multiplication, size %dx%d, total time: %.5f seconds" % 
      (rows, cols, total_time))

rows = 2000
cols = rows
A = np.random.rand(rows, cols)
B = np.random.rand(rows, cols)

start_time = time.time_ns()
result = np.dot(A, B)
end_time = time.time_ns()
#print(result)

total_time = (end_time-start_time) / 1000000000
print("np.dot, size %dx%d, total time: %.5f seconds" % 
      (rows, cols, total_time))

"""
Output on my computer: 
My multiplication, size 500x500, total time: 42.48177 seconds
np.dot, size 2000x2000, total time: 0.12253 seconds
My function was about 347 times slower on a much smaller problem.

Output on Google Colab: 
My multiplication, size 500x500, total time: 55.68286 seconds
np.dot, size 2000x2000, total time: 0.52767 seconds
np.dot, size 500x500, total time: 0.00958 seconds
My function was about 5800 times slower than np.dot for size 500x500.
"""

#%%
