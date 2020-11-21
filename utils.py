import sys
import math
import random
import numpy as np  # used in generating dataset and plotting results
from itertools import cycle
from matplotlib import pyplot as plt


def generate_data(num_points: int, dimension: int, god_separation: list, lower_bound: int, upper_bound: int,
                  data_path=None):
    """Generate data with requested format and characteristic
    Args:
        num_points: number of data points to generate
        dimension: number of dimension for data points
        god_separation: a list indicating a perfect separation plane for the data points
        lower_bound: all dimension values are not smaller than it
        upper_bound: all dimension values are smaller than it
        data_path: indicating the path to write the dataset
    """
    if data_path is None:
        data_path="source.txt"
    with open(data_path, 'w') as f:
        print(f"{num_points} {dimension}", file=f)
        for _ in range(num_points):
            line_of_point = np.random.randint(low=lower_bound, high=upper_bound, size=dimension).tolist()
            while np.dot(line_of_point, god_separation) == 0:  # ensure that there are no points on god_separation
                line_of_point = np.random.randint(low=lower_bound, high=upper_bound, size=dimension).tolist()
            line_of_point.append(1 if np.dot(line_of_point, god_separation) > 0 else 0)
            print(str(line_of_point).replace(',', '').strip('[]'), file=f)
    return data_path


def update_GUI(ln, d, w, lb, ub):
    """Update the current plane found
    Args:
        ln: graph representation of the current plane
        d: number of dimension for data points
        w: a numeric list indicating the current plane
        lb: lower bound
        ub: upper bound
    Returns:
        ln: the updated ln
    """
    if ln is not None:
        ln.remove()
    # check if w[1] or w[2] is 0
    if d == 2:
        if w[1] == 0:
            ln, = plt.plot([0, 0], [lb, ub])  # plot line based on new w
        else:
            ln, = plt.plot([lb, ub], [-lb * w[0] / w[1], -ub * w[0] / w[1]])
    else:
        if w[2] == 0 and w[1] == 0:
            y, z = np.meshgrid(np.arange(lb, ub, 5), np.arange(lb, ub, 5))
            x = z * 0
        elif w[2] == 0 and w[0] == 0:
            x, z = np.meshgrid(np.arange(lb, ub, 5), np.arange(lb, ub, 5))
            y = z * 0
        else:
            x, y = np.meshgrid(np.arange(lb, ub, 5), np.arange(lb, ub, 5))
            z = (w[0] * x + w[1] * y) / -w[2]
        ln = plt.gcf().gca(projection='3d').plot_surface(x, y, z, alpha=0.3, linewidth=0, antialiased=False)
    plt.pause(0.01)
    return ln


def plot_dataset(data_path: str, sample_size=None):
    """
    accept the file path and plot points based on the data set
    :param data_path: the path of the file that contains all the data
    :return: n - number of points, d - dimensions, ub - upper bound, lb - lower bound
    """
    ub = -sys.maxsize - 1
    lb = sys.maxsize
    fig = plt.figure()
    x, y, z, label = [], [], [], []
    
    with open(data_path, 'r') as f:
        raw = f.readlines()
        n, d = list(map(int, raw[0].strip().split()))
        if sample_size is None and n > 1000:
            data = random.sample(raw[1:], 1000)
        elif sample_size is not None:
            data = random.sample(raw[1:], sample_size)
        else:
            data = raw[1:]
        for index in range(len(data)):
            point = list(map(int, data[index].strip().split()))
            ub = max(max(point[:-1]), ub)
            lb = min(min(point[:-1]), lb)
            if d > 3:
                return n, d, None, None
            x.append(point[0])
            y.append(point[1])
            if d == 3:
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
    return n, d, ub, lb


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
    
    if ((len(w) + 1) != len(point)):
        raise Exception(f"Length of w:{w} + 1 not equal to point:{point}")
    result = 0
    for i in range(len(w)):
        result += w[i] * point[i]
    if result >= 0 and point[-1] == 0:
        return True
    elif result <= 0 and point[-1] == 1:
        return True
    else:
        return False


def get_iter(data_path: str):
    """
    Args:
        data_path: the path where dataset locates
    Returns:
        An iterator yielding the data points
    """
    for line in open(data_path, 'r'):
        yield line.strip()


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

    def update_w(w, point):
        sign = 1 if point[-1] == 1 else -1
        for i in range(len(w)):
            w[i] += sign * point[i]
    
    last_index = -1
    num_violations = 0
    threshold = 5000
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
            if d <= 3 and num_violations % 1000 == 1:
                ln = update_GUI(ln, d, w, lb, ub)
            if num_violations >= threshold:  # forced-termination
                threshold *= 2
                num_violations = 0
                w = [0] * d
            last_index = index
        elif index % n == last_index % n:  # self-termination
            if d <= 3:
                ln = update_GUI(ln, d, w, lb, ub)
            break
    return w