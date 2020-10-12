import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']


def get_data(list_paths):
    data = {}
    for list_path in list_paths:
        with open(list_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            _, gender, age = line.replace("\n", "").split(',')
            if int(age) > 90 or int(age) < 10: continue
            if age in data.keys():
                data[age] = data[age] + 1
            else:
                data[age] = 1
    keys = sorted(data.keys())
    data_size = [data[k] for k in keys]
    return keys, data_size


def main():
    data_class, data_size = get_data(["dataset/agedb_list.txt",
                                      "dataset/megaage_asian_list.txt",
                                      "dataset/afad_list.txt"])
    print(data_class)
    print(data_size)
    plt.bar(range(len(data_size)), data_size, color='rgb', tick_label=data_class)
    index = sorted(range(len(data_class)), reverse=False)
    for a, b in zip(index, data_size):
        plt.text(a, b + 1, b, ha='center', va='bottom')
    plt.xlabel('年龄')
    plt.ylabel('数量')
    plt.title('年龄分布直方图')
    plt.show()


if __name__ == '__main__':
    main()
