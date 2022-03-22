import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from path_config import wine_path, wine_file_list, oakland_path, oakland_file_list


# 创建文件夹
def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Data():
    def __init__(self):
        self.read_data = [os.path.join("data", wine_path), os.path.join("data", oakland_path)]
        self.write_data = [os.path.join("result", wine_path), os.path.join("result", oakland_path)]
        self.data_file_list = [wine_file_list, oakland_file_list]

    # 特征处理
    def process_features(self):
        for i in range(len(self.data_file_list)):
            for file_name in self.data_file_list[i]:
                content = pd.read_csv(os.path.join(self.read_data[i], file_name))  # 单个的csv读取数据DataFrame
                write_data_path = os.path.join(self.write_data[i], file_name.split('.')[0])
                print("*" * 50)
                print("Begin to process file: %s" % file_name)
                for title in content.columns.values:
                    if title == "Unnamed: 0":  # 未命名0列
                        continue
                    if content[title].dtypes == "int64" or content[title].dtypes == "float64":
                        self.process_num_features(content, title, write_data_path)  # 处理数值属性
                    else:
                        self.process_nom_features(content, title, write_data_path)   # 处理标称属性

    # 处理标称属性
    def process_nom_features(self, content, title, write_data_path):
        write_file_path = os.path.join(write_data_path, 'nominal_attribute')
        makedir(write_file_path)
        value_dict = self.get_feature_value(content, title)
                
        with open(os.path.join(write_file_path, title+".txt"), "w", encoding='utf-8') as fp:
            fp.write(("Feature Name: %s\n" % title))
            fp.write(("Value Num: %s\n" % len(value_dict)))
            for i in value_dict:
                fp.write(str(i) + "," + str(value_dict[i]) + "\n")
        
        print("*" * 50)
        print("Feature Name: %s" % title)
        print("Value Num: %s" % len(value_dict))

    # 处理数值属性
    def process_num_features(self, content, title, write_data_path):
        write_file_path = os.path.join(write_data_path, 'numeric_attribute')
        makedir(write_file_path)

        max_num = content[title].max()  # 最大值
        min_num = content[title].min()  # 最小值
        mean_num = content[title].mean()  # 平均值
        median_num = content[title].median()   # 中位数
        quartile1 = content[title].quantile(0.25)  # 四分位数
        quartile2 = content[title].quantile(0.75)
        missing_num = len(content) - content[title].count()    # 缺失数

        with open(os.path.join(write_file_path, title+".txt"), "w") as fp:
            fp.write(("Feature Name: %s\n" % title))
            fp.write(("Max Num: %s\n" % max_num))
            fp.write(("Min Num: %s\n" % min_num))
            fp.write(("Mean Num: %s\n" % mean_num))
            fp.write(("Median Num: %s\n" % median_num))
            fp.write(("Quartile Num: %s, %s\n" % (quartile1, quartile2)))
            fp.write(("Missing Num: %s\n" % missing_num))

        print("*" * 50)
        print("Feature Name: %s" % title)
        print("Max Num: %s" % max_num)
        print("Min Num: %s" % min_num)
        print("Mean Num: %s" % mean_num)
        print("Median Num: %s" % median_num)
        print("Quartile Num: %s, %s" % (quartile1, quartile2))
        print("Missing Num: %s" % missing_num)

        write_figure_path = os.path.join(write_data_path, 'figure')
        makedir(write_figure_path)
        self.draw_figure(content, title, write_figure_path)
       
    def draw_figure(self, content, title, write_figure_path):
        # 绘制直方图
        figure_path = os.path.join(write_figure_path, title + "_histogram.png")
        self.draw_histogram(content, title, figure_path)
        # 绘制盒图
        figure_path = os.path.join(write_figure_path, title + "_box.png")
        self.draw_box(content, title, figure_path)

    # 绘制直方图
    def draw_histogram(self, content, title, write_figure_path):
        content = content.dropna(subset=[title])
        plt.hist(content[title], 20)
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Freq")
        plt.savefig(write_figure_path)
        plt.close()

    # 绘制盒图
    def draw_box(self, content, title, write_figure_path):
        content = content.dropna(subset=[title])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.boxplot(content[title], sym="o", whis=1.5, labels=[title])
        plt.savefig(write_figure_path)
        plt.close()

    # 统计标称属性的数量
    def get_feature_value(self, content, title):
        value_dict = dict()
        for i in range(len(content)):
            if pd.isnull(content[title][i]):
                continue
            if content[title][i] in value_dict:
                value_dict[content[title][i]] += 1
            else:
                value_dict[content[title][i]] = 1
        return value_dict

    # 数据填充
    def filling(self):
        for name in range(len(self.data_file_list)):
            for file_name in self.data_file_list[name]:
                content = pd.read_csv(os.path.join(self.read_data[name], file_name))
                write_data_path = os.path.join(self.write_data[name], file_name.split('.')[0])
                print("*" * 50)
                print("Begin to filling file: %s" % file_name)

                # 填充策略1：剔除缺失的内容
                strategy_path = os.path.join(write_data_path, "strategy_1")
                makedir(strategy_path)
                with open(os.path.join(strategy_path, file_name), 'w', encoding='utf-8') as fp1:
                    strategy1_content = content
                    for title in content.columns.values:
                        if title == "Unnamed: 0":
                            strategy1_content = strategy1_content.drop(columns=[title])
                        elif strategy1_content[title].dtypes == "int64" or strategy1_content[title].dtypes == "float64":
                            strategy1_content = strategy1_content.dropna(subset=[title])
                            self.draw_figure(strategy1_content, title, strategy_path)

                # 填充策略2：用最高频率值来填补缺失值
                strategy_path = os.path.join(write_data_path, "strategy_2")
                makedir(strategy_path)
                with open(os.path.join(strategy_path, file_name), 'w', encoding='utf-8') as fp2:
                    strategy2_content = content
                    for title in content.columns.values:
                        if title == "Unnamed: 0":
                            strategy2_content = strategy2_content.drop(columns=[title])
                        elif strategy2_content[title].dtypes == "int64" or strategy2_content[title].dtypes == "float64":
                            value_dict = self.get_feature_value(strategy2_content, title)
                            filling_data = max(value_dict, key=value_dict.get)
                            strategy2_content = strategy2_content.fillna({title:filling_data})
                            self.draw_figure(strategy2_content, title, strategy_path)

                # 通过属性的相关关系来填补缺失值
                strategy_path = os.path.join(write_data_path, "strategy_3")
                makedir(strategy_path)
                with open(os.path.join(strategy_path, file_name), 'w', encoding='utf-8') as fp3:
                    strategy3_content = content
                    strategy3_content = strategy3_content.interpolate(kind='nearest')
                    for title in content.columns.values:
                        if title == "Unnamed: 0":
                            continue
                        elif strategy3_content[title].dtypes == "int64" or strategy3_content[title].dtypes == "float64":
                            self.draw_figure(strategy3_content, title, strategy_path)

                # 通过数据对象之间的相似性来填补缺失值
                strategy_path = os.path.join(write_data_path, "strategy_4")
                makedir(strategy_path)
                with open(os.path.join(strategy_path, file_name), 'w', encoding='utf-8') as fp4:
                    strategy4_content = content
                    nonan_content = pd.DataFrame()
                    num_list = []
                    for title in strategy4_content.columns.values:
                        if title == "Unnamed: 0":
                            strategy4_content = strategy4_content.drop(title, 1)
                        elif strategy4_content[title].dtypes == "int64" or strategy4_content[title].dtypes == "float64": 
                            num_list.append(title)
                            nonan_content = pd.concat([nonan_content, strategy4_content[title]], axis=1)
                    nonan_content.dropna(axis=0, how='any', inplace=True)
                    mean_val = [nonan_content[title].mean() for title in nonan_content.columns.values]
 
                    if len([title for title in num_list if strategy4_content[title].isnull().any() == True]) == len(num_list):
                        for i in range(len(strategy4_content)):
                            if strategy4_content.loc[i][num_list].isnull().all():
                                for j in range(len(num_list)):
                                    strategy4_content.loc[i, num_list[j]] = mean_val[j]

                    for title in num_list: 
                        if strategy4_content[title].isnull().any():
                            train_y1 = nonan_content[title]
                            train_x1 = nonan_content.loc[:, [other for other in num_list if other!=title]]
                            test_x1 = strategy4_content[pd.isna(strategy4_content[title])].loc[:, [other for other in num_list if other!=title]]
                            index, pred = self.knn_missing_filled(train_x1, train_y1, test_x1)
                            strategy4_content.loc[index, title] = pred
                        self.draw_figure(strategy4_content, title, strategy_path)

    # 使用k近邻算法，计算填充内容
    def knn_missing_filled(self, x_train, y_train, test, k=3, dispersed=True):
        if dispersed:
            clf = KNeighborsClassifier(n_neighbors=k, weights="distance")
        else:
            clf = KNeighborsRegressor(n_neighbors=k, weights="distance")
        
        clf.fit(x_train, y_train)
        return test.index, clf.predict(test)


if __name__ == "__main__":
    data = Data()

    data.process_features()

    data.filling()


