import numpy as np

def calculate_average(data):
    return np.average(data)

def calculate_variance(data):
    return np.var(data, ddof=1)

def calculate_pool_variance(data):
    sum = 0
    n = 0
    k = 0
    for key in data:
        sum += (data[key]['length'] - 1) * data[key]['variance']
        n += data[key]['length']
        k += 1
    return sum/(n-k)

def calculate_f_statistic(data, g_avg, pool_variance):
    sum = 0
    k = 0
    for key in data:
        sum += data[key]['length']*(data[key]['average'] - g_avg)**2
        k += 1
    if k-1 == 0 or pool_variance == 0:
          return(np.inf)
    return (sum/(k-1))/pool_variance


def get_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            row = line[:-1].split(',')
            data.append(row)
    return data

def select_feature(n, f_scores):
    return sorted(range(len(f_scores)), key=lambda i: f_scores[i])[-n:]


def f_test(raw_data):
    Y = np.array(raw_data[0],dtype='float')
    X = np.array(raw_data[1:],dtype='float')

    class_ids = set(Y)
    F_test_scores = []
    for row in X:
        row_summary = {}
        for id in class_ids:
            class_data = []
            row_summary[id] = {}
            for i in range(len(row)):
                if Y[i] == id:
                    class_data.append(row[i])

            row_summary[id]['length'] = len(class_data)
            row_summary[id]['average'] = calculate_average(class_data)
            row_summary[id]['variance'] = calculate_variance(class_data)
            # if row_summary[id]['variance'] == 0:
            #     print(class_data)

        # print(row_summary)
        pool_variance = calculate_pool_variance(row_summary)
        row_average = calculate_average(row)
        F_score = calculate_f_statistic(row_summary, row_average, pool_variance)
        F_test_scores.append(F_score)
        # print(F_score)

    # print(F_test_scores)
    top_feature_numbers = select_feature(100, F_test_scores)

    top_feature_scores=[]
    for i in top_feature_numbers:
        top_feature_scores.append([F_test_scores[i],i])

    top_feature_scores = (sorted(top_feature_scores)[::-1])
    return top_feature_scores,F_test_scores

def print_scores(top_feature_numbers,F_test_scores):
        for i in top_feature_numbers:
            print(i)

if __name__ == '__main__':
#    test = [[1,1,1,1,2,1,1,2,1,2],[125,100,70,120,95,60,220,85,75,90]]
    file_name = 'HandWrittenLetters.txt'
    raw_data = get_data(file_name)
    features, scores = f_test(raw_data)
    print_scores(features, scores)




