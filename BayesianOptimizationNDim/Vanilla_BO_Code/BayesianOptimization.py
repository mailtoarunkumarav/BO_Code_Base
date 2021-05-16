import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Base Class for running Bayesian Optmization Core
class BayesianOptimization:

    # Initialize the parameters of Bayesian Optimization Object
    def __init__(self, name, gp_object, acq_func_obj, func_helper_obj, no_iterations, plot_final_posterior_distr, plot_for_iterations):

        self.name = name
        self.gp_obj = gp_object
        self.acq_func_obj = acq_func_obj
        self.no_iterations = no_iterations
        self.func_helper_obj = func_helper_obj
        self.y_true_max = func_helper_obj.get_true_max()
        self.plot_final_posterior_distr = plot_final_posterior_distr
        self.plot_for_iterations = plot_for_iterations


    def run_bayes_opt(self, run_count):

        # print the parameters for the current run
        print("************  Run: " + str(
            run_count) + "     ACQ:" + self.acq_func_obj.acq_type.upper() + "    ***************\n\n")
        print('Initial Values for this run\n---X---\n', self.gp_obj.X)
        print('---y---\n', self.gp_obj.y, '\n\n')

        # generate the points X* for the function evaluations
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

        # # plot visuals before optimization, used when debugging each posterior distr in the 1D case
        # print ("Before Optimization" )
        # mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
        # standard_deviation = np.sqrt(diag_variance)
        # self.gp_obj.plot_visuals(Xs, ys, mean, standard_deviation, f_prior, f_post)
        # self.gp_obj.plot_posterior_predictions(10, Xs, ys, mean, standard_deviation)

        regret = [[]]
        print("Starting Optimization")

        # Run the optimization for the specified number of iterations and thus finding the best suggestions to observe the function
        for i in range(self.no_iterations):
            print("######  run: " + str(run_count) + "  iteration: ", i + 1, "  ######")

            # Optimizing the characteristic length scale for every 3 iterations (it might be expensive if estimated at every iteration)
            if self.gp_obj.params_estimation and (i + 1) % 3 == 0:
                # Set the initial values and parameters required for the maximiser
                init_var = self.gp_obj.signal_variance
                init_len_scale = self.gp_obj.char_length_scale
                init_points = np.append(init_len_scale, init_var)
                total_bounds = [self.gp_obj.len_scale_bounds for nd in np.arange(self.gp_obj.number_of_dimensions)]
                total_bounds.append(self.gp_obj.signal_variance_bounds)
                print("Initial length scale: ", init_len_scale)
                print("Initial variance: ", init_var )
                maxima = opt.minimize(lambda x: -self.gp_obj.optimize_log_marginal_likelihood(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      bounds= total_bounds)

                self.gp_obj.char_length_scale = maxima['x'][: self.gp_obj.number_of_dimensions]
                self.gp_obj.signal_variance = maxima['x'][len(maxima['x']) -1]
                print("Opt Length scale: ", self.gp_obj.char_length_scale, "\nOpt variance: ", self.gp_obj.signal_variance)
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

            print(xnew, "is the new value added..")

            # Add the new observation point to the existing set of observed samples along with its true value
            X = self.gp_obj.X
            X = np.append(X, [xnew], axis=0)

            # calculate the true function value and add it to the existing set of values
            ynew = self.func_helper_obj.get_true_func_value(np.matrix(xnew))
            y = np.append(self.gp_obj.y, ynew, axis=0)

            # Refit the GP model to use the updated prior knowledge
            self.gp_obj.gaussian_fit(X, y)

            # # plots for detailed analysis
            mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
            standard_deviation = np.sqrt(diag_variance)

            if i > 5 and i% self.plot_for_iterations == 0:
                self.gp_obj.plot_posterior_predictions(self.acq_func_obj.acq_type + "_"+str(run_count) + '_iteration:' + str(i + 1), Xs, ys,
                                                       mean, standard_deviation)

            # plot_axes = [self.gp_obj.linspacexmin, self.gp_obj.linspacexmax, -10, 10]
            # self.acq_func_obj.plot_acquisition_function(str(run_count) + '_' + str(i + 1), Xs, np.diagonal(acq_func_values), plot_axes)

            # Calculate the regret after each iteration as the difference of the maximum value observed and the true
            # maximum in the given bounds

            ith_regret = self.y_true_max - self.gp_obj.y.max()
            regret = np.append(regret, np.matrix(ith_regret))
            print("\n")

        # Display the final values for this iteration
        print('Final values:\n$X: \n', self.gp_obj.X.T, '\n$y:\n', self.gp_obj.y.T)
        print("True Max: ", self.y_true_max)
        print(self.acq_func_obj.acq_type.upper(),", Observed Maximum value: ", self.gp_obj.y.max())
        print("Regret: ", regret)
        print("\n\n\n")

        if self.plot_final_posterior_distr:
            print("Plotting after optimization")
            # plot visuals after optimization in each run
            # self.plot_regret(regret, self.no_iterations)
            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = self.gp_obj.gaussian_predict(Xs)
                standard_deviation = np.sqrt(diag_variance)
            self.gp_obj.plot_visuals(self.acq_func_obj.acq_type.upper() + "_Final" + str(run_count), Xs, ys, mean, standard_deviation,
                                     f_prior,
                                     f_post)


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


