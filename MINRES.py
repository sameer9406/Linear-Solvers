import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags
from scipy.sparse import linalg as lg



# Quadratic function
class Quad(object):
    def __init__(self, Q, p):
        self.Q = Q
        self.p = p

    def func(self, x):
        r_0 = self.p - np.dot(self.Q, x)
        return r_0


# Solver
class MINRES(object):

    def __init__(self, A, b, x_0, TOL, MAXIT):
        self.func = Quad(A, b)
        self.TOL = TOL
        self.MAXIT = MAXIT
        self.Q = self.func.Q
        self.x = x_0
        self.r_vec = [self.func.func(self.x)]
        self.p_vec = [self.r_vec[-1]]
        self.Qr_vec = [np.dot(self.Q, self.r_vec[-1])]
        self.res_vec = [np.linalg.norm(self.r_vec[-1])]
        self.Qp = np.dot(self.Q, self.p_vec[-1])
        self.k = 1

    def calc(self, ):
        while self.res_vec[-1] > self.TOL and self.k < self.MAXIT:
            alpha = np.divide(np.dot(self.r_vec[-1].T, self.Qr_vec[-1]), np.dot(self.Qp.T, self.Qp))
            self.x += alpha * self.p_vec[-1]
            self.r_vec.append(self.r_vec[-1] - (alpha * self.Qp))
            self.Qr_vec.append(np.dot(self.Q, self.r_vec[-1]))
            self.res_vec.append(np.linalg.norm(self.r_vec[-1]))
            beta = np.divide(np.dot(self.r_vec[-1].T, self.Qr_vec[-1]), np.dot(self.r_vec[-2].T, self.Qr_vec[-2]))
            self.p_vec.append(self.r_vec[-1] + np.dot(beta, self.p_vec[-1]))
            self.Qp = np.dot(self.Q, self.p_vec[-1])
            self.k += 1
        return self.res_vec, self.k, self.x


# Generate Matrix
class gen_mat(object):

    def __init__(self, n):
        self.n = n

    def gen_diag(self, ):
        data = np.array([- np.ones(self.n), 2 * np.ones(self.n), - np.ones(self.n)])
        diags = np.array([-1, 0, 1])
        M = spdiags(data, diags, self.n, self.n).toarray()
        return M


def main(n):
    mat = gen_mat(n)
    run = MINRES(mat.gen_diag(), -np.ones(n), np.zeros(n), 1e-5, 2000)
    sci = lg.minres(mat.gen_diag(), -np.ones(n), np.zeros(n), tol=1e-5, maxiter=2000)
    result = run.calc()
    print('result :', result[2])
    print('result_sci :', sci[0])
    plt.semilogy(range(0, result[1]), np.log(result[0]), label='MINRES')
    plt.legend()
    plt.ylabel('Residuals')
    plt.xlabel('Iterations')
    plt.show()


if __name__ == '__main__':
    main(25)
