## 数据集
        Wine Reviews
        Oakland Crime Statistics 2011 to 2016
## 运行环境
        python 3.6
        pandas、matplotlib、sklearn
## 运行方法
        python data_handle.py

        # 修改在path_config.py中个人的数据集路径
## 文件说明
        data_handle.py

        # 文件的主程序，包括数据的读写、填充、绘图等主要方法
        
        path_config.py

        # 程序需要处理的数据的路径配置
        
        data文件夹
        
        # 包括需要处理的数据:oakland-crime-statistics-2011-to-2016、wine-reviews
        
        result文件夹
        
        # figure，正常数据下生成的直方图和盒图
        # nominal_attribute，标称属性的统计数据
        # numeric_attribute，数值属性的数值计算结果：最大值、最小值、平均值、中位数、四分位数、缺失数据       
        # strategy_1,将缺失部分剔除生成的直方图和盒图
        # strategy_2,用最高频率值来填补缺失值生成的直方图和盒图
        # strategy_3,通过属性的相关关系来填补缺失值生成的直方图和盒图
        # strategy_4,通过数据对象之间的相似性来填补缺失值生成的直方图和盒图
