import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import linalg as lg

# Quadratic function
class Quad(object):
    def __init__(self, Q, p):
        self.Q = Q
        self.p = p

    def grad_func(self, x):
        df = np.dot(self.Q, x) - self.p
        return df


# Solver
class CG_Solve(object):

    def __init__(self, A, b, x_0, TOL, MAXIT):
        self.func = Quad(A, b)
        self.x = x_0
        self.TOL = TOL
        self.MAXIT = MAXIT
        self.g = self.func.grad_func(self.x)
        self.d = - self.g
        self.k = 1
        self.res = [np.linalg.norm(self.g)]

    def calc(self, ):
        while self.res[-1] > self.TOL and self.k < self.MAXIT:
            z = np.dot(self.func.Q, np.copy(self.d))
            alpha = np.linalg.norm(self.g) ** 2 / np.dot(self.d.T, z)
            self.x += alpha * self.d
            self.g += alpha * z
            res = np.linalg.norm(self.g)
            self.res.append(res)
            beta = (self.res[-1] / self.res[-2]) ** 2
            self.d = - self.g + beta * self.d
            self.k += 1
        return self.res, self.k, self.x


# Generate Matrix
class gen_mat(object):

    def __init__(self, n):
        self.n = n

    def gen_diag(self, ):
        data = np.array([- np.ones(self.n), 2 * np.ones(self.n), - np.ones(self.n)])
        diags = np.array([-1,0,1])
        M = spdiags(data, diags, self.n, self.n).toarray()
        return M

def main(n):
    mat = gen_mat(n)
    run = CG_Solve(mat.gen_diag(), -np.ones(n),np.zeros(n), 1e-5, 2000)
    sci = lg.cg(mat.gen_diag(), -np.ones(n), np.zeros(n), 1e-5, 2000)
    result = run.calc()
    print('result :', result[2])
    print('result_sci :', sci[0])
    plt.semilogy(range(0, result[1]), np.log(result[0]), label='CG')
    plt.legend()
    plt.ylabel('Residuals')
    plt.xlabel('Iterations')
    plt.show()


if __name__ == '__main__':
    main(25)
