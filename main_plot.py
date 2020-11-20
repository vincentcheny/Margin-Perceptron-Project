import sys
import getopt
import os
import numpy as np  # only used for generating dataset
from itertools import tee, cycle
from matplotlib import pyplot as plt


def plot_dataset(data_path: str):
    """
    accept the file path and plot points based on the data set
    :param data_path: the path of the file that contains all the data
    :return: n - number of points, d - dimensions, ub - upper bound, lb - lower bound
    """
    ub = -sys.maxsize - 1
    lb = sys.maxsize
    fig = plt.figure()
    x, y, z, label = [], [], [], []
    for index, line in enumerate(open(data_path, 'r')):
        if index == 0:
            info = list(map(int, line.strip().split()))
        else:
            point = list(map(int, line.strip().split()))
            ub = max(max(point[:-1]), ub)
            lb = min(min(point[:-1]), lb)
            if info[1] > 3:
                return info[1], None, None
            x.append(point[0])
            y.append(point[1])
            if info[1] == 3:
                z.append(point[2])
            label.append(point[-1])
    if not z:
        plt.scatter(x, y, c=list(map(lambda x: 'r' if x == 1 else 'b', label)))
        plt.xlim(lb - 2, ub + 2)
        plt.ylim(lb - 2, ub + 2)
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c=list(map(lambda x: 'r' if x == 1 else 'b', label)))
        ax.set_xlim(lb - 2, ub + 2)
        ax.set_ylim(lb - 2, ub + 2)
        ax.set_zlim(lb - 2, ub + 2)
    return info[0], info[1], ub, lb


def is_violated(w: list, point: list):
    """Judge if a point is a violation point

    Args:
        w: a list, namely the values of estimate plane in every dimension
        point: a list, namely the values in every dimension and a label value

    Returns:
        A Bool value: Violation -> True, no violation -> false

    Raises:
        IOError: An error occurred accessing w or point, or dimensionality is different

    """
    result = 0
    if ((len(w) + 1) != len(point)):
        raise Exception
    for i in range(len(w)):
        result += w[i] * point[i]
    if result >= 0 and point[-1] == 0:
        return True
    elif result <= 0 and point[-1] == 1:
        return True
    else:
        return False


def generate_data(num_points: int, dimension: int, god_separation: list, lower_bound: int, upper_bound: int,
                  data_path="source.txt"):
    """Generate data with requested format and characteristic

    Args:
        num_points: number of data points to generate
        dimension: number of dimension for data points
        god_separation: a list indicating a perfect separation plane for the data points
        lower_bound: all dimension values are not smaller than it
        upper_bound: all dimension values are smaller than it
        data_path: indicating the path to write the dataset
    """
    with open(data_path, 'w') as f:
        print(data_path)
        print(f"{num_points} {dimension}", file=f)
        for _ in range(num_points):
            line_of_point = np.random.randint(low=lower_bound, high=upper_bound, size=dimension).tolist()
            while np.dot(line_of_point, god_separation) == 0:  # ensure that there are no points on god_separation
                line_of_point = np.random.randint(low=lower_bound, high=upper_bound, size=dimension).tolist()
            line_of_point.append(1 if np.dot(line_of_point, god_separation) > 0 else 0)
            print(str(line_of_point).replace(',', '').strip('[]'), file=f)
    return data_path


def get_iter(data_path: str):
    """
    Args:
        data_path: the path where dataset locates

    Returns:
        An iterator yielding the data points
    """
    for line in open(data_path, 'r'):
        yield line.strip()


def update_w(w, point):
    sign = 1 if point[-1] == 1 else -1
    for i in range(len(w)):
        w[i] += sign * point[i]


def main_loop(data_iter, w, n=None, d=None, ub=None, lb=None):
    """Generate data with requested format and characteristic

    Args:
        num_points: number of data points to generate
        dimension: number of dimension for data points
        god_separation: a list indicating a perfect separation plane for the data points
        lower_bound: all dimension values are not smaller than it
        upper_bound: all dimension values are smaller than it
        data_path: indicating the path to write the dataset

    Returns:
        An iterator yielding the data points
    """

    def square_sum(iterator):
        return sum(list(map(lambda x: int(x) ** 2, iterator.split())))

    data_iter, data_iter_sum = tee(data_iter)
    data_iter_sum.__next__()
    R = max(list(map(square_sum, data_iter_sum)))
    gamma = R
    last_index = -1
    num_violations = 0

    ln = None

    for index, line in enumerate(cycle(data_iter)):
        line = line.strip().split()
        if n is None or d is None or index < n:  # handle the first iteration of all data
            point = list(map(int, line))
            if index == 0:
                n, d = point
                w = [0] * d
                continue
            assert len(point) == d + 1, f"The read line:{point} has incorrect number of dimensions."
        elif isinstance(line, list):  # data format is sure to be correct after an iteration
            point = list(map(int, line))
        else:
            raise TypeError(f"line is in an invalid type: {line}.")
        if index % (n + 1) == 0:  # skip the title line
            continue
        if is_violated(w, point):
            num_violations += 1
            update_w(w, point)

            # update GUI
            if d <= 3:
                if ln is not None:
                    ln.remove()
                if d == 2:
                    ln, = plt.plot([lb, ub], [-lb * w[0] / w[1], -ub * w[0] / w[1]])  # plot line based on new w
                else:
                    x = np.arange(lb, ub, 5)
                    y = np.arange(lb, ub, 5)
                    x, y = np.meshgrid(x, y)
                    z = (w[0] * x + w[1] * y) / -w[2]
                    ln = plt.gcf().gca(projection='3d').plot_surface(x, y, z, alpha=0.3, linewidth=0, antialiased=False)
                plt.pause(0.1)  # update gui

            if num_violations >= 12 * R * R / gamma / gamma:  # forced-termination
                num_violations = 0
                gamma /= 2
                w = [0] * d
            last_index = index
        elif index % n == last_index % n:  # self-termination
            break
    return w, gamma


if __name__ == "__main__":
    input_path, output_path = None, None
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:", ["input_file=", "output_file="])
    except getopt.GetoptError:
        print('main.py -i <input_file> -o <output_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('main.py -i <input_file> -o <output_file>')
            sys.exit()
        elif opt in ("-i", "--input_file"):
            input_path = arg
        elif opt in ("-o", "--output_file"):
            output_path = arg

    w = None  # result of our algorithm

    if input_path == None or os.path.isfile(input_path) == False:
        use_a_fixed_config_to_test = True
        if use_a_fixed_config_to_test:
            n = 100
            # d=6
            # god_separation = [-1,-1,-1,-1,-1,-1]
            d = 3
            god_separation = [-1, -1, -1]
            lb = -100
            ub = 100
        else:
            print(f"Dataset {input_path} not detected. Use a new generated dataset.")
            input_str = input(
                "Please enter the number of points and dimensions, separated by a space (e.g. \'15 3\'):\n")
            if len(input_str.split()) == 2 or all(i.isdigit() and int(i) > 0 for i in input_str.split()):
                pass
            else:
                raise ValueError("Number of points and dimensions are not valid.")
            n = int(input_str.split()[0])
            d = int(input_str.split()[1])

            print(f"Please enter god_separation with {d} dimensions:")
            god_separation = list(map(int, input().split()))
            assert len(god_separation) == d, f"Length of god_separation is not equal to {d}."

            print("Please enter [lower bound, upper bound) of points (e.g. \'-3 7\'):")
            lb, ub = list(map(int, input().split()))
            assert lb < ub, f"The input lower bound and upper bound are not valid."
        output_path = generate_data(n, d, god_separation, lower_bound=lb, upper_bound=ub, data_path=output_path)
        plot_dataset(output_path)  # plot points from data set
        data_iter = get_iter(output_path)
        w, gamma = main_loop(data_iter, w, n, d, ub, lb)  # added two parameters ub lb
        if d <= 3:
            plt.show()  # keep the gui window open
    else:
        n, d, ub, lb = plot_dataset(input_path)   # plot data set points and get n, d, ub, lb
        with open(input_path) as data_iter:
            w, gamma = main_loop(data_iter, w, n, d, ub, lb)  # added 4 parameters n, d, ub, lb
        if d <= 3:
            plt.show()  # keep the gui window open

    print(f"Final result: {w} with gamma {gamma}")
