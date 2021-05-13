import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm

# Base Class for running Bayesian Optmization Core
class BayesianOptimization:

    # Initialize the parameters of Bayesian Optimization Object
    def __init__(self, name, gp_object, acq_func_obj, func_helper_obj, no_iterations):

        self.name = name
        self.gp_obj = gp_object
        self.acq_func_obj = acq_func_obj
        self.no_iterations = no_iterations
        self.func_helper_obj = func_helper_obj
        # self.y_true_max = func_helper_obj.get_true_max()
        self.y_true_max = None



    def func(self,x):

        sum =0
        for item in x:
            sum += item
        value = np.exp(-x[0]) * np.sin(10*np.pi*x[0])
        return value

    def run_bayes_opt(self, run_count):

        # print the parameters for the current run
        print("************  Run: " + str(
            run_count) + "     ACQ:" + self.acq_func_obj.acq_type.upper() + "    ***************\n\n")
        print('Initial Values for this run\n---X---\n', self.gp_obj.X)
        print('---y---\n', self.gp_obj.y, '\n\n')

        # generate the points Xstar for the function evaluations
        random_points = []
        Xs = []

        # Generate specified (number of unseen data points) random numbers for each dimension
        for dim in np.arange(self.gp_obj.number_of_dimensions):
            random_data_point_each_dim = np.linspace(self.gp_obj.bounds[dim][0], self.gp_obj.bounds[dim][1],
                                                     self.gp_obj.number_of_test_datapoints).reshape(1,
                                                                                                    self.gp_obj.number_of_test_datapoints)
            random_points.append(random_data_point_each_dim)
        random_points = np.vstack(random_points)

        # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
        for sample_num in np.arange(self.gp_obj.number_of_test_datapoints):
            array = []
            for dim_count in np.arange(self.gp_obj.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            Xs.append(array)
        Xs = np.vstack(Xs)

        # Obtain the values for the true function, so that true function can be plotted in the case of 1D currently
        ys = self.func_helper_obj.get_true_func_value(Xs)

        ###Normalizing code
        Xs = np.divide((Xs - self.gp_obj.Xmin), (self.gp_obj.Xmax - self.gp_obj.Xmin))
        # #normalizing y
        ys = (ys - self.gp_obj.ymin) / (self.gp_obj.ymax - self.gp_obj.ymin)

        # # plot visuals before optimization, used when debugging each posterior distr in the 1D case
        # print ("Before Optimization" )
        # mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
        # standard_deviation = np.sqrt(diag_variance)
        # # # self.gp_obj.plot_posterior_predictions(10, Xs, ys, mean, standard_deviation)
        # self.gp_obj.plot_posterior_predictions('_' + self.gp_obj.kernel_char +'_' , Xs, ys, mean, standard_deviation)

        # Boolean to keep track of the function evaluations at X =[0], so that genuine vales are being added
        zero_value_bool = False
        regret = [[]]
        print("Starting Optimization")

        # Run the optimization for the number of iterations and hence finding the best points to observe the function
        for i in range(self.no_iterations):
            print("######  run: " + str(run_count) + "  iteration: ", i + 1,"    " ,self.gp_obj.kernel_char,"  ######")

            ## Optimizing the characteristic length scale
            if self.gp_obj.len_scale_estimation and (i+1) % 1 == 0:

                # Estimating Length scale itself
                x_max_value = None
                log_like_max = - 1* float("inf")

                # Data structure to create the starting points for the scipy.minimize method
                random_points = []
                starting_points = []

                # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
                for dim in np.arange(self.gp_obj.number_of_dimensions):
                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_bounds[dim][0],
                                                                   self.gp_obj.len_scale_bounds[dim][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).\
                                                                    reshape(1,self.gp_obj.number_of_restarts_likelihood)
                    random_points.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points = np.vstack(random_points)

                # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
                for sample_num in np.arange(self.gp_obj.number_of_restarts_likelihood):
                    array = []
                    for dim_count in np.arange(self.gp_obj.number_of_dimensions):
                        array.append(random_points[dim_count, sample_num])
                    starting_points.append(array)
                starting_points = np.vstack(starting_points)

                variance_start_points = np.random.uniform(self.gp_obj.signal_variance_bounds[0],
                                                                   self.gp_obj.signal_variance_bounds[1],
                                                                   self.gp_obj.number_of_restarts_likelihood)

                total_bounds = self.gp_obj.len_scale_bounds.copy()
                total_bounds.append(self.gp_obj.signal_variance_bounds)

                for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):

                    init_len_scale = starting_points[ind]
                    init_var = variance_start_points[ind]

                    init_points = np.append(init_len_scale, init_var)
                    # print("Initial length scale: ", init_len_scale, "\nInitial variance: ", init_var)
                    maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_l(x),
                                          init_points,
                                          method='L-BFGS-B',
                                          tol=0.01,
                                          # options={'maxfun': 2000, 'maxiter': 2000},
                                          bounds=total_bounds)

                    len_scale_temp = maxima['x'][:self.gp_obj.number_of_dimensions]
                    variance_temp = maxima['x'][len(maxima['x']) - 1]
                    params = np.append (len_scale_temp, variance_temp)
                    log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_l(params)

                    if (log_likelihood > log_like_max ):
                        print("New maximum log likelihood ", log_likelihood, " found for l= ",
                              maxima['x'][: self.gp_obj.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])
                        x_max_value = maxima
                        log_like_max = log_likelihood

                self.gp_obj.char_length_scale = x_max_value['x'][:self.gp_obj.number_of_dimensions]
                self.gp_obj.signal_variance = x_max_value['x'][len(maxima['x']) - 1]

                print("Opt Length scale: ", self.gp_obj.char_length_scale, "\nOpt variance: ", self.gp_obj.signal_variance)
                # Recomputing L according to the updated length scale
                self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)

            ## Optimizing the characteristic length scale parameters : ***** working for 1D only
            if (self.gp_obj.params_estimation and (i + 1) % 1 == 0):
                # # #Testing here
                # lens = np.linspace(0.0001, 100 ,1000)
                # val = []
                # max = - 1* float("inf")
                # for each_len_scale in lens:
                #      temp = self.gp_obj.optimize_log_marginal_likelihood_l_params(np.array([each_len_scale]))
                #      # print(each_len_scale,"---", temp)
                #      if(temp>max):
                #         max_obj = str(each_len_scale)+"--"+str(temp)
                #         max = temp
                #      val.append(np.asarray(temp)[0])
                # print("max values ", max_obj)
                # # print(lens.shape, len(val))
                # self.acq_func_obj.plot_graph(str(run_count) + '_' + str(i + 1), lens,val, [0,10000,0,15])
                #####
                x_max_value = None
                log_like_max = - 1* float("inf")

                # print("*****************************Debugging here********************************")
                # self.gp_obj.number_of_dimensions = 3
                # self.gp_obj.len_scale_func_type = ['quadratic', 'linear', 'quadratic']
                # self.gp_obj.len_scale_param_bounds = [ [[0,3],[3, 4.0001],[5, 6.0001]], [[7,8],[8, 9.0001]], [[9,10],[10, 11.0001],[11,12]]]
                # self.gp_obj.number_of_restarts_likelihood = 30

                random_points_a = []
                random_points_b = []
                random_points_c = []

                for d in np.arange(self.gp_obj.number_of_dimensions):
                    # Data structure to create the starting points for the scipy.minimize method
                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][0][0],
                                                                   self.gp_obj.len_scale_param_bounds[d][0][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                   self.gp_obj.number_of_restarts_likelihood)
                    random_points_a.append(random_data_point_each_dim)


                    random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][1][0],
                                                                   self.gp_obj.len_scale_param_bounds[d][1][1],
                                                                   self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                   self.gp_obj.number_of_restarts_likelihood)
                    random_points_b.append(random_data_point_each_dim)

                    if self.gp_obj.len_scale_func_type[d] == 'linear' or self.gp_obj.len_scale_func_type[d] == 'gaussian' :
                        # Two Parameters by default
                        pass

                    elif self.gp_obj.len_scale_func_type[d] == 'quadratic' :

                        random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[d][2][0],
                                                                       self.gp_obj.len_scale_param_bounds[d][2][1],
                                                                       self.gp_obj.number_of_restarts_likelihood).reshape(1,
                                                                       self.gp_obj.number_of_restarts_likelihood)
                        random_points_c.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points_a = np.vstack(random_points_a)
                random_points_b = np.vstack(random_points_b)

                variance_start_points = np.random.uniform(self.gp_obj.signal_variance_bounds[0],
                                                          self.gp_obj.signal_variance_bounds[1],
                                                          self.gp_obj.number_of_restarts_likelihood)

                if 'quadratic' in self.gp_obj.len_scale_func_type:
                    random_points_c = np.vstack(random_points_c)

                for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):

                    tot_init_points = []
                    total_bounds = []
                    count=0
                    for each_dimension in np.arange(self.gp_obj.number_of_dimensions):
                        param_a = random_points_a[each_dimension][ind]
                        tot_init_points.append(param_a)
                        param_b = random_points_b[each_dimension][ind]
                        tot_init_points.append(param_b)
                        total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][0])
                        total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][1])

                        if self.gp_obj.len_scale_func_type[each_dimension] == "linear" \
                                or self.gp_obj.len_scale_func_type[each_dimension]  == "gaussian":
                            pass

                        elif self.gp_obj.len_scale_func_type[each_dimension] == "quadratic":
                            param_c = random_points_c[count][ind]
                            tot_init_points.append(param_c)
                            count= count+1
                            total_bounds.append(self.gp_obj.len_scale_param_bounds[each_dimension][2])

                    tot_init_points.append(variance_start_points[ind])
                    total_bounds.append(self.gp_obj.signal_variance_bounds)

                    maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_l_params(x),
                                              tot_init_points,
                                              method='L-BFGS-B',
                                              tol=0.01,
                                              # options={'maxfun': 1000, 'maxiter': 1000},
                                              bounds=total_bounds)

                    params = maxima['x']
                    log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_l_params(params)
                    if (log_likelihood > log_like_max):
                        print("New maximum log likelihood ", log_likelihood, " found for params ", params)
                        x_max_value = maxima['x']
                        log_like_max = log_likelihood


                total_len_scale_params = []
                param_count = 0
                for type in self.gp_obj.len_scale_func_type:
                    if(type == 'linear') or (type == 'gaussian'):
                        total_len_scale_params.append(x_max_value[param_count:(param_count+2)])
                        param_count += 2
                    elif(type == 'quadratic'):
                        total_len_scale_params.append(x_max_value[param_count:(param_count+3)])
                        param_count += 3

                self.gp_obj.len_scale_params = total_len_scale_params
                self.gp_obj.signal_variance = x_max_value[len(maxima['x']) - 1]

                print("Opt Params: ", self.gp_obj.len_scale_params, "   variance:", self.gp_obj.signal_variance)

                # Recomputing L according to the updated length scale
                self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)

            # Random search to find the maxima in the specified bounds
            if (self.acq_func_obj.acq_type == 'rs'):
                print("performing RS for the maxima")

                # Randomly select the values for Xnew and perform random search for optima in the search space
                xnew = np.array([])
                for dim in np.arange(self.gp_obj.number_of_dimensions):
                    value = np.random.uniform(self.gp_obj.bounds[dim][0], self.gp_obj.bounds[dim][1], 1).reshape(1, 1)
                    xnew = np.append(xnew, value)

            else:
                # Maximising the acquisition function to obtain the value at which the function has to be evaluated next
                xnew, acq_func_values = self.acq_func_obj.max_acq_func(self.gp_obj, Xs, ys, i + 1)


            # Notify if zeroes are being added as the best point to evaluate next
            if (np.array_equal(xnew, np.zeros(self.gp_obj.number_of_dimensions))):
                zero_value_bool = True
                print('\nzeroes encountered in run: ', run_count, " iteration: ", i + 1)

            xnew_orig = np.multiply(xnew.T, (self.gp_obj.Xmax - self.gp_obj.Xmin)) + self.gp_obj.Xmin
            # #xnew_orig = xnew * (self.gp_obj.Xmax - self.gp_obj.Xmin) + self.gp_obj.Xmin
            ynew_orig = self.func_helper_obj.get_true_func_value(np.matrix(xnew_orig))

            # Add the new observation point to the existing set of observed samples along with its true value
            X = self.gp_obj.X
            X = np.append(X, [xnew], axis=0)

            # calculate the true function value and add it to the existing set of values
            # ynew = self.func_helper_obj.get_true_func_value(np.matrix(xnew))
            # # # Normalizing
            ynew = (ynew_orig - self.gp_obj.ymin)/ (self.gp_obj.ymax - self.gp_obj.ymin)


            # Commenting ynew augment for standardizing logic implemented above
            y = np.append(self.gp_obj.y, ynew, axis=0)
            print("(",xnew,ynew, ") is the new value added..    Original: ", (xnew_orig, ynew_orig) )

            # Refit the GP model to use the updated prior knowledge
            self.gp_obj.gaussian_fit(X, y)

            # #commented as it is required only for debugging
            # #plot posterior after each iteration
            # mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
            # standard_deviation = np.sqrt(diag_variance)
            # if(self.gp_obj.number_of_dimensions==1
            #         # and i == self.no_iterations-1
            # ):
            #     self.gp_obj.plot_posterior_predictions(self.acq_func_obj.acq_type + str(run_count) + '_' + str(i + 1)+
            #                                            "_"+self.gp_obj.kernel_char,Xs, ys, mean, standard_deviation)
                # plot_axes = [self.gp_obj.linspacexmin, self.gp_obj.linspacexmax, 0, 1]
                # self.acq_func_obj.plot_acquisition_function(str(run_count) + '_' + str(i + 1), Xs, np.diagonal(acq_func_values), plot_axes)
            ######################

            # # plot acq functions and posteriors if its required to verify [0] being added to X
            # if(zero_value_bool):
            #     mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
            #     standard_deviation = np.sqrt(diag_variance)
            #     # self.gp_obj.plot_posterior_predictions(str(run_count)+'_'+str(i + 1), Xs, ys, mean, standard_deviation)
            #     plot_axes = [self.gp_obj.linspacexmin, self.gp_obj.linspacexmax, 0, 20]
            #     self.acq_func_obj.plot_acquisition_function(i+1, Xs, acq_func_values, plot_axes)
            #     zero_value_bool =False


            # Calculate the regret after each iteration as the difference of the maximum value observed and the true
            # maximum in the given bounds

            # y_de_std = np.multiply(self.gp_obj.y, np.std(self.gp_obj.y)) + np.mean(self.gp_obj.y)
            ith_regret = self.y_true_max  - y.max()
            regret = np.append(regret, np.matrix(ith_regret))
            print("\n")

        # Display the final values for this iteration
        print('Final values:\n$X: \n', self.gp_obj.X.T, '\n$y:\n', self.gp_obj.y.T)
        print("True Max: ", self.y_true_max)
        print(self.acq_func_obj.acq_type.upper(), self.gp_obj.kernel_char,", Observed Maximum value: ", self.gp_obj.y.max())
        print("Regret: ", regret)
        print("\n\n\n")

        # print("After Optimization")
        # # plot visuals after optimization for each iteration
        # self.plot_regret(regret, self.no_iterations)
        # with np.errstate(invalid='ignore'):
        #     mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
        #     standard_deviation = np.sqrt(diag_variance)
        # self.gp_obj.plot_visuals(run_count, Xs, ys, mean, standard_deviation, f_prior, f_post)

        return regret

    def plot_regret(self, regret, iterations):

        # Plot the regret for each of the iteration
        iterations_axes = np.arange(start=1, stop=iterations + 1, step=1)
        plt.figure("Regret ")
        plt.clf()
        plt.plot(iterations_axes, regret, 'b')
        plt.axis([1, iterations, 0, 1])
        plt.title('Regret for iterations: ' + str(iterations))
        plt.savefig('regret.png')



'''

                #############################

                # Estimating Length scale itself
                # x_max_value = None
                # log_like_max = - 1* float("inf")
                #
                # # Data structure to create the starting points for the scipy.minimize method
                # random_points_a = []
                # random_points_b = []
                #
                # # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
                # for dim in np.arange(self.gp_obj.number_of_dimensions):
                #     random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[dim][0],
                #                                                    self.gp_obj.len_scale_param_bounds[dim][1],
                #                                                    self.gp_obj.number_of_restarts_likelihood). reshape(1,
                #                                                    self.gp_obj.number_of_restarts_likelihood)
                #     random_points_a.append(random_data_point_each_dim)
                #
                # # Vertically stack the arrays of randomly generated starting points as a matrix
                # random_points_a = np.vstack(random_points_a)
                #
                # # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
                # for dim in np.arange(self.gp_obj.number_of_dimensions):
                #     random_data_point_each_dim = np.random.uniform(self.gp_obj.len_scale_param_bounds[dim][0],
                #                                                    self.gp_obj.len_scale_param_bounds[dim][1],
                #                                                    self.gp_obj.number_of_restarts_likelihood). \
                #         reshape(1, self.gp_obj.number_of_restarts_likelihood)
                #     random_points_b.append(random_data_point_each_dim)
                #
                # # Vertically stack the arrays of randomly generated starting points as a matrix
                # random_points_b = np.vstack(random_points_b)
                #
                # total_bounds = self.gp_obj.len_scale_param_bounds.copy()
                #
                # for ind in np.arange(self.gp_obj.number_of_restarts_likelihood):
                #
                #     param_a = random_points_a[0][ind]
                #     param_b = random_points_b[0][ind]
                #
                #     init_points = np.append(param_a, param_b)
                #     # print("Initial values for a & b are ",param_a, param_b)
                #     maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood_l_params(x),
                #                           init_points,
                #                           method='L-BFGS-B',
                #                           bounds=total_bounds)
                #
                #     a_temp = maxima['x'][0]
                #     b_temp = maxima['x'][1]
                #     params = np.append(a_temp, b_temp)
                #     log_likelihood = self.gp_obj.optimize_log_marginal_likelihood_l_params(params)
                #
                #     if (log_likelihood > log_like_max):
                #         print("New maximum log likelihood ", log_likelihood, " found for params ", params)
                #         x_max_value = maxima
                #         log_like_max = log_likelihood
                #
                # a = x_max_value['x'][0]
                # b = x_max_value['x'][1]
                # self.gp_obj.len_scale_params[0] = [a,b]
                #
                # print("Opt Params: ", self.gp_obj.len_scale_params)
                #
                # # Recomputing L according to the updated length scale
                # self.gp_obj.L_x_x = self.gp_obj.compute_l(self.gp_obj.X)

                #######################


'''