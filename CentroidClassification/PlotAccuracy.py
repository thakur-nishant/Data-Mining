import matplotlib.pyplot as plt
import random

import CentroidClassification as cc


def run():
    # filename = input("Enter filename: ")
    filename = "HandWrittenLetters.txt"

    # class_ids = [1,2,3,4,5,6,7,8,9,10]
    class_ids = random.sample(range(1, 27), 10)
    # class_ids = [x for x in range(11, 21)]
    print("For class:", class_ids)

    n = 38
    i = 5
    x_axis = []
    y_axis = []
    while i < n:
        test_instances = [i, n]
        print("\n############################")
        print("When train =", i, "and test =", n - i + 1)
        accuracy = cc.start(filename, class_ids, test_instances)
        x_axis.append(i)
        y_axis.append(accuracy)
        # plt.scatter(i, accuracy, s=80, c="b", marker="x")
        i += 5

    plt.plot(x_axis, y_axis)
    plt.show()


if __name__ == "__main__":
    run()
