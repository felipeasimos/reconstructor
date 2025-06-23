import numpy as np

def lu_decompose(A):
    L = np.identity(A.shape[0])
    U = np.zeros(A.shape)
    n = A.shape[0]
    for k in range(n):
        for j in range(k, n):
            U[k][j] = A[k][j] - np.sum(L[k, :k] * U[:k, j])
        for i in range(k+1, n):
            L[i][k] = ( A[i][k] - np.sum(L[i, :k] * U[:k, k]) )/U[k][k]
    return L, U

def forward_substitution(L, b):
    y = np.zeros(b.shape)
    n = L.shape[0]
    for i in range(n):
        y[i] = (b[i] - np.sum(L[i,:i] * y[:i]))/L[i][i]
    return y

def backwards_substitution(U, y):
    x = np.zeros(y.shape)
    n = U.shape[0]
    for i in range(n-1,-1,-1):
        x[i] = (y[i] - np.sum(U[i,i+1:n] * x[i+1:]))/U[i][i]
    return x

def lu_solve(A, Y):
    L, U = lu_decompose(A)
    y = forward_substitution(L, Y)
    return backwards_substitution(U, y)

# LU decomposition, for solving the linear system

def get_x_sum(saved_sums, points, power):
    if(saved_sums[power] is None):
        s = 0
        for x, _ in points:
            s+=x**power
        saved_sums[power] = s
    return saved_sums[power]

def get_xy_sum(points, power):
    s = 0
    if(power==0):
        for _, y in points:
            s+= y
    else:
        for x, y in points:
            s+= y * x**power
    return s

def polynomial_fit_matrices(points, polynomial_degree):
    n = polynomial_degree+1
    saved_x_sums=[None]*(2*n-1)
    A = np.empty((n, n))
    Y = np.empty((n))
    for i in range(n):
        for j in range(n):
            A[i][j] = get_x_sum(saved_x_sums, points, i+j)
        Y[i] = get_xy_sum(points, i)

    #print("A:\n", A)
    #print("Y:\n", Y)
    return A, Y

def polynomial_get_coefficients(points, polynomial_degree, solve_method=None):
    A, Y = polynomial_fit_matrices(points, polynomial_degree)
    if(solve_method is None):
        solve_method = np.linalg.solve
    return solve_method(A, Y)

def polynomial(coefficients, x):
    s = 0
    for i in range(len(coefficients)):
        s += coefficients[i] * x**i
    return s

if( __name__ == "__main__" ):

    import matplotlib.pyplot as plt
    from matplotlib.collections import PathCollection
    from matplotlib.backend_bases import MouseEvent

    class EditablePolynomial:
        def __init__(self, points, degree, solve_method=None):
            self.points = points
            self.drawing_points : list[PathCollection] = []
            self.degree = degree
            self.coefficients = []
            self.solve_method = solve_method

            # define xs where are going to compute and the limits
            # of the plot
            x_max = np.array(self.points).max(axis=0)[0] + 10
            x_min = np.array(self.points).min(axis=0)[0] - 10
            delta = ( x_max - x_min )/250
            self.graph_x = [ x_min + ( delta * i) for i in range(251) ]

            y_min = min([ point[1] for point in points ]) - 10
            y_max = max([ point[1] for point in points ]) + 10

            self.press = False
            self.connect()
            self.draw()

            plt.ylim([ y_min, y_max ])
            plt.xlim([ x_min, x_max ])
            plt.autoscale(enable=False)
            plt.show()

        def clean(self):
            if(len(self.graph)):
                self.graph[0].remove()

        def draw(self):

            self.coefficients = polynomial_get_coefficients(self.points, self.degree)
            #print("coefficients (from smaller degree to greater):\n", self.coefficients)
            graph_y = [ polynomial(self.coefficients, x) for x in self.graph_x ]
            
            self.graph = plt.plot(self.graph_x, graph_y, color='blue')
            self.figure.canvas.draw()

        def connect(self):

            figure, ax = plt.subplots()

            for point in self.points:
                self.drawing_points.append(ax.scatter(point[0], point[1], color='red'))
            figure.canvas.mpl_connect('button_press_event', self.on_press)
            figure.canvas.mpl_connect('button_release_event', self.on_release)
            figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

            self.figure = figure

        def on_press(self, event : MouseEvent):
            # check if a point has been pressed
            for i, drawing_point in enumerate(self.drawing_points):
                if drawing_point.contains(event)[0]:
                    self.press = True
                    self.prepare_point_change(i)
                    break

        def on_release(self, event : MouseEvent):
            self.press = False
            self.update_point([ event.xdata, event.ydata ])
            self.clean()
            self.draw()

        def on_motion(self, event : MouseEvent):
            if self.press:
                self.update_point([ event.xdata, event.ydata ])
                self.clean()
                self.draw()

        def update_point(self, new_coords):
            self.drawing_points[-1].set_offsets( new_coords )
            self.points[-1] = new_coords

        def prepare_point_change(self, index : int):

            if( index != len(points) - 1 ):
                tmp = self.points[index]
                self.points[index] = self.points[-1]
                self.points[-1] = tmp

                tmp = self.drawing_points[index]
                self.drawing_points[index] = self.drawing_points[-1]
                self.drawing_points[-1] = tmp
    #
    # points = np.array([[0, 0],
    #                     [1, 2],
    #                     [2, 3],
    #                     [3, 1]], dtype=np.float64)
    points = np.array([[13, 0.804],
                       [14, 1.716],
                       [15, 3.617],
                       [16, 7.710]], dtype=np.float64)
    degree = int(input("Input the degree of the polynomial: "))
    if(degree > len(points)-1):
        print("given polynomial degree is too high for the quantity of points given.")
        print("please provide more points or choose a smaller degree.")
    else:
        EditablePolynomial(points, degree, lu_solve)
