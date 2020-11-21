import sys
import getopt
import os
import time
from matplotlib import pyplot as plt
from utils import generate_data, plot_dataset, get_iter, main_loop


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
    st = time.time()
    print("Running...")

    if input_path == None or os.path.isfile(input_path) == False:
        print(f"No input dataset or input dataset not detected.")
        use_given_config = input("Generate the dataset with customized config? (1/0):")
        if use_given_config == '0':
            n = 20000
            d = 3
            god_separation = [2, -1, 2]
            lb = -10000
            ub = 10000
        elif use_given_config == '1':
            input_str = input(
                "Please enter the number of points and dimensions, separated by a space (e.g. \'100 3\'):\n")
            if len(input_str.split()) == 2 or all(i.isdigit() and int(i) > 0 for i in input_str.split()):
                pass
            else:
                raise ValueError("Number of points and dimensions are not valid.")
            n = int(input_str.split()[0])
            d = int(input_str.split()[1])

            print(f"Please enter god_separation with {d} dimensions:")
            god_separation = list(map(int, input().split()))
            assert len(god_separation) == d, f"Length of god_separation is not equal to {d}."

            print("Please enter [lower bound, upper bound) of points (e.g. \'-100 100\'):")
            lb, ub = list(map(int, input().split()))
            assert lb < ub and lb * ub < 0, f"The input lower bound {lb} and upper bound {ub} are not valid."
        else:
            raise ValueError(f"Input {use_given_config} is not valid.")
        output_path = generate_data(n, d, god_separation, lower_bound=lb, upper_bound=ub, data_path=output_path)
        plot_dataset(output_path)  # plot points from data set
        data_iter = get_iter(output_path)
        w = main_loop(data_iter, w, n, d, ub, lb)  # added two parameters ub lb
        print(f"Finish in {time.time()-st:.2f}s.")
        print(f"Tartget plane: {god_separation}")
    else:
        n, d, ub, lb = plot_dataset(input_path)
        print(f"Dataset size: {n} dimensions: {d}")
        with open(input_path) as data_iter:
            w = main_loop(data_iter, w, n, d, ub, lb)  # added 4 parameters n, d, ub, lb
        print(f"Finish in {time.time()-st:.2f}s.")
    print(f"Found separation plane: {w}")
    if d <= 3:
        plt.show()  # keep the gui window open
    
        