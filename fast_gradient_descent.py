import numpy as np
import matplotlib.pyplot as plt


class BS(object):

    def grad_func(x):
        return np.array([(4 * x[0] - 2) ** 3 + 2 * (x[0] - 2 * x[1]), -4 * (x[0] - 2 * x[1])])


class fast_grad(object):

    def __init__(self, x_0):
        self.TOL = 10 ** -6
        self.MAXIT = 50
        self.x_0 = x_0
        self.alpha_0 = 10 ** -16
        self.theta_0 = 10 ** 16
        self.k = 1
        self.res = []
        self.k_a = [1]
        self.res.append(1)
        self.x_1= self.x_0 - self.alpha_0 * BS.grad_func(x_0)




    def calc(self, ):
        while self.res[-1] > self.TOL and self.k < self.MAXIT:
            alpha_k = min(np.sqrt(1 + self.theta_0) * self.alpha_0, np.linalg.norm(self.x_1 - self.x_0) / (2 * np.linalg.norm(
                              BS.grad_func(self.x_1) - BS.grad_func(self.x_0))))
            theta_k = alpha_k / self.alpha_0
            x_k = self.x_1 - self.alpha_0 * BS.grad_func(self.x_1)
            self.r = np.linalg.norm(BS.grad_func(x_k))
            self.res.append(self.r)
            self.x_0 = self.x_1
            self.x_1 = x_k
            self.alpha_0 = alpha_k
            self.theta_0 = theta_k
            self.k += 1
            self.k_a.append(self.k)
        return self.res, self.k_a, x_k




def main():
    run = fast_grad(np.array([1.5, 1.5]))
    result = run.calc()
    print('result :', result[2])
    plt.plot(result[1], result[0])
    plt.show()


if __name__ == '__main__':
    main()
