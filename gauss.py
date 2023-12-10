import numpy as np

def gauss(A, b):
    # erwiterte Koeffizientenmatrix erstellen
    lgs = np.column_stack((A, b))
    n = len(b)

    for i in range(n):
        # Pivotzeile suchen/tauschen
        pivot_index = np.argmax(np.abs(lgs[i:, i])) + i
        lgs[[i, pivot_index], :] = lgs[[pivot_index, i], :]

        # Normalisierung
        lgs[i, :] = lgs[i, :] / lgs[i, i]

        # Elimination
        for j in range(i+1, n):
            lgs[j, :] = lgs[j, :] - lgs[i, :] * lgs[j, i]

    # Rücksubstitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = lgs[i, -1] - np.dot(lgs[i, i+1:n], x[i+1:])
    return x

A = np.array([[2, 4, 1], [3, 6, -2], [4, 6, 3]], dtype=np.float)
b = np.array([9, -1, 13], dtype=np.float)

result = gauss(A, b)
print(result)
