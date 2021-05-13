import numpy as np

class Custom_Print(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self) :
        for f in self.files:
            f.flush()

class FunctionHelper:

    def __init__(self, func_type):
        self.true_func_type = func_type

    def get_true_max(self):

        # define y_max for the true functions
        if (self.true_func_type == 'custom'):
            # exp{-(x-2)^2} + exp{-((x-6)^2)/10} + (1/(X^2 +1))
            # true_max = self.get_true_func_value(2.0202)

            # exp(-x)sin(3x) + 0.3
            # true_max = self.get_true_func_value(0.15545)

            # exp(-x)sin(8.pi.x) + 1
            # true_max = self.get_true_func_value(0.061)

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            #in the range [0.5, 2.5] max is 1.253 @x = 0.3479
            true_max = self.get_true_func_value( 0.3479 )

            # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0 @x=1.0
            # true_max = self.get_true_func_value(1.0)

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima = 0.7887), -exp(-x)*sin(2.pi.x) (minima)
            # true_max = self.get_true_func_value(0.22488)

        elif (self.true_func_type == 'sin'):
            true_max = self.get_true_func_value(1.57079)

        elif (self.true_func_type == 'branin2d'):
            # self.y_true_max = 0.397887
            true_max = self.get_true_func_value(np.matrix([[9.42478, 2.475]]))

        elif (self.true_func_type == 'sphere'):
            # self.y_true_max = 0
            true_max = self.get_true_func_value(np.matrix([[0, 0]]))

        elif (self.true_func_type == 'hartmann3d'):
            # self.y_true_max = 3.86278
            # x = [0.114614, 0.555649, 0.852547]
            true_max = self.get_true_func_value(np.matrix([[0.114614, 0.555649, 0.852547]]))

        elif (self.true_func_type == 'hartmann6d'):
            # self.y_true_max = 3.32237
            # x = [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]
            true_max = self.get_true_func_value(np.matrix([[0.20169, 0.150011, 0.476874, 0.275332,
                                                                   0.311652, 0.6573]]))
        print("True function:",self.true_func_type," \nMaximum is ", true_max)
        return true_max

    # function to evaluate the true function depending on the selection
    def get_true_func_value(self, x):
        if (self.true_func_type == 'sin'):
            return np.sin(x)

        elif (self.true_func_type == 'cos'):
            return np.cos(x)

        elif (self.true_func_type == 'custom'):
            # exp{-(x-2)^2} + exp{-((x-6)^2)/10} + (1/(X^2 +1))
            # return np.exp(-(x - 2) ** 2) + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)

            # exp(-x)sin(3.pi.x) + 0.3
            # return (np.exp(-x) * np.sin(3 * np.pi * x)) + 0.3

            # exp(-x)sin(8.pi.x) + 1
            # return (np.exp(-x) * np.sin(8 * np.pi * x)) + 1

            # Gramacy and Lee function sin(10.pi.x/2x)+(x-1)^4; minima = -2.874 @ x=0.144; -sin(10.pi.x/2x)-x-1)^4; maxima = 2.874 @x=0.144
            return ((((np.sin(10 * np.pi * x))/(2*(x)))) +(x-1) ** 4) * -1

            # # Levy function w = 1+(x-1)/4  y = (sin(w*pi))^2 + (w-1)^2(1+(sin(2w*pi))^2) max =0
            # # # w = -0.5+((x-1)/4)
            # w = 1+((x-1)/4)
            # value = ((np.sin(w * np.pi))**2 + ((w-1)**2)*(1+((np.sin(2*w*np.pi)) ** 2 )))
            # return -1 * value

            # Benchmark Function exp(-x)*sin(2.pi.x)(maxima), -exp(-x)*sin(2.pi.x) (minima)
            # return (np.exp(-x) * np.sin(2 * np.pi * x))


        elif (self.true_func_type == 'branin2d'):
            # branin 2d fucntion
            # a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π)
            # y = a * (x2 - b * x1 **2 + c * x1 - r) ** 2 + s * (1 - t) * cos(x1) + s
            x1 = x[:, 0]
            x2 = x[:, 1]
            a = 1;
            b = 5.1 / (4 * (np.pi ** 2));
            c = 5 / np.pi;
            r = 6;
            s = 10;
            t = 1 / (8 * np.pi)
            value = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
            value = -1 * value.reshape((-1, 1))
            return value

        elif (self.true_func_type == 'sphere'):
            # simple sphere equation
            # Z = X**2 + Y**2
            x1 = x[:, 0]
            x2 = x[:, 1]
            value = (x1 ** 2 + x2 ** 2)
            value = -1 * value.reshape(-1, 1)
            return value

        elif (self.true_func_type == 'hartmann3d'):

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A_array = [[3, 10, 30],
                       [0.1, 10, 35],
                       [3, 10, 30],
                       [0.1, 10, 35]
                       ]
            A = np.matrix(A_array)

            P_array = [[3689, 1170, 2673],
                       [4699, 4387, 7470],
                       [1091, 8732, 5547],
                       [381, 5743, 8828]
                       ]

            P = np.matrix(P_array)
            P = P * 1e-4

            sum = 0
            for i in np.arange(0, 4):
                alpha_value = alpha[i]
                inner_sum = 0
                for j in np.arange(0, 3):
                    inner_sum += A.item(i, j) * ((x[:, j] - P.item(i, j)) ** 2)
                sum += alpha_value * np.exp(-1 * inner_sum)
            # extra -1 is because we are finding maxima instead of the minima f(-x)
            value = (-1 * -1 * sum).reshape(-1, 1)
            return value

        elif (self.true_func_type == 'hartmann6d'):

            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            A_array = [[10, 3, 17, 3.50, 1.7, 8],
                       [0.05, 10, 17, 0.10, 8, 14],
                       [3, 3.5, 1.7, 10, 17, 8],
                       [17, 8, 0.05, 10, 0.1, 14]
                       ]
            A = np.matrix(A_array)

            P_array = [[1312, 1696, 5569, 124, 8283, 5886],
                       [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650],
                       [4047, 8828, 8732, 5743, 1091, 381]
                       ]
            P = np.matrix(P_array)
            P = P * 1e-4

            sum = 0
            for i in np.arange(0, 4):
                alpha_value = alpha[i]
                inner_sum = 0
                for j in np.arange(0, 6):
                    inner_sum += A.item(i, j) * ((x[:, j] - P.item(i, j)) ** 2)
                sum += alpha_value * np.exp(-1 * inner_sum)
            # extra -1 is because we are finding maxima instead of the minima f(-x)
            value = (-1 * -1 * sum).reshape(-1, 1)
            return value


