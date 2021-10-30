import numpy
import pandas
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def main():
    # 加载数据
    data = pd.DataFrame(pd.read_csv("./data.csv", skiprows=1, names=[0,1,2]))

    # 分离特征与标签
    x_all_feature = data.iloc[:, 0:2]
    y_all_tag = data.iloc[:, 2:]

    # 划分训练集与测试集
    x_train_feature, x_test_feature, y_train_tag, y_test_tag = train_test_split(x_all_feature, y_all_tag, test_size=0.2, random_state=520)

    # 分离训练集和测试集
    x1_all_feature, x2_all_feature = data.iloc[:, 0:1], data.iloc[:, 1:2]
    x1_train_feature, x2_train_feature = x_train_feature.iloc[:, 0:1], x_train_feature.iloc[:, 1:2]
    x1_test_feature, x2_test_feature = x_test_feature.iloc[:, 0:1], x_test_feature.iloc[:, 1:2]


    # print(logistic_regression_model_1.coef_)

    # plt.subplot(3, 3, 1)
    # plt.scatter(data.loc[data[2]>0, 0], data.loc[data[2]>0, 1],c='red',s=50,cmap="rainbow")#rainbow彩虹色

    # plt.subplot(3, 3, 1)
    # plt.scatter(data.loc[data[2]<0, 0], data.loc[data[2]<0, 1],c='black',s=50,cmap="rainbow")#rainbow彩虹色

    # (a)(i) 画原始数据
    # plt.subplot(1, 1, 1)
    # plt.rc('font', size=18)
    # compare_in_map(plt=plt, com_pre=data)

    # plt.xlabel('x_1')
    # plt.ylabel('x_2')

    # 原始数据
    plt.figure(figsize=(9, 9))
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    # plt.legend(['+1_original', '-1_originsssal'])
    com_pre = data
    feature_number = 2

    plt.scatter(com_pre.loc[com_pre[feature_number] > 0, 0], com_pre.loc[com_pre[feature_number] > 0, 1],
                label='+1_original',
                s=30, c='r', marker='+', alpha=0.8, linewidths=1, edgecolors=None)
    plt.scatter(com_pre.loc[com_pre[feature_number] < 0, 0], com_pre.loc[com_pre[feature_number] < 0, 1],
                label='-1_original',
                s=30, c='g', marker='o', alpha=0.8, linewidths=0, edgecolors=None)


    plt.rcParams['figure.constrained_layout.use'] = True
    plt.legend()

    plt.figure(figsize=(9, 9))
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    # plt.legend(['+1_original', '-1_originsssal'])
    com_pre = data
    feature_number = 2
    plt.scatter(com_pre.loc[com_pre[feature_number]>0, 0], com_pre.loc[com_pre[feature_number]>0, 1],
                label='+1_original',
                s = 30, c = 'r', marker = '+', alpha = 0.8, linewidths = 1, edgecolors = None)
    plt.scatter(com_pre.loc[com_pre[feature_number]<0, 0], com_pre.loc[com_pre[feature_number]<0, 1],
                label='-1_original',
                s = 30, c = 'g', marker = 'o', alpha = 0.8, linewidths = 0, edgecolors = None)

    # compare_in_map(plt=plt, com_pre=data)
                   # s1 = 3, c1 = c1, marker1 = marker1, alpha1 = alpha1, linewidths1 = linewidths1, edgecolors1 = edgecolors1,
                   # s2 = 3, c2 = c2, marker2 = marker2, alpha2 = alpha2, linewidths2 = linewidths2, edgecolors2 = edgecolors2)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # compare_in_map(plt=plt, com_pre=data,
    #                xlabel = '', ylable = '', legend = None,
    #                s1 = 3, c1 = c1, marker1 = marker1, alpha1 = alpha1, linewidths1 = linewidths1, edgecolors1 = edgecolors1,
    #                s2 = 3, c2 = c2, marker2 = marker2, alpha2 = alpha2, linewidths2 = linewidths2, edgecolors2 = edgecolors2)




    # (a)(ii)
    # 实例化逻辑回归模型
    logistic_regression_model_1 = LR(C=1, max_iter=1000)
    # 训练模型
    logistic_regression_model_1 = logistic_regression_model_1.fit(x_all_feature, y_all_tag)
    # 输出模型参数
    print("(a)(ii) 逻辑回归参数:")
    print("coef_:" + str(logistic_regression_model_1.coef_))
    print("intercept_:" + str(logistic_regression_model_1.intercept_))

    # (a)(iii) 逻辑回归模型的预测表现
    # logistic_regression_model_1 = logistic_regression_model_1.fit(x_train_feature, y_train_tag)
    # Z = clf.decision_function(xy).reshape(axisx.shape)
    y_tag_predited = pd.DataFrame(logistic_regression_model_1.predict(x_all_feature))
    x_t = x_all_feature.reset_index(drop=True)
    com_pre = pd.concat([x_t, y_tag_predited],axis=1)
    com_pre.columns = [0, 1, 2]

    # 画逻辑回归预测散点图
    # plt.subplot(1, 1, 1)
    # compare_in_map(plt=plt, com_pre=com_pre,
    #                legend=['+1_original','-1_original','+1_logistic regression','-1_logistic regression'])


    # plt.legend()
    # plt.xlabel('x_1')
    # plt.ylabel('x_2')
    # plt.legend(['+1_original','-1_original','+1_logistic regression','-1_logistic regression'],)
    # plt.legend()
    feature_number = 2
    plt.scatter(com_pre.loc[com_pre[feature_number]>0, 0], com_pre.loc[com_pre[feature_number]>0, 1],
                label='+1_logistic regression',
                s = 20, c = 'm', marker = '+', alpha = 0.8, linewidths = 1, edgecolors = '#000000')
    plt.scatter(com_pre.loc[com_pre[feature_number]<0, 0], com_pre.loc[com_pre[feature_number]<0, 1],
                label='-1_logistic regression',
                s = 20, c = '#FF99CC', marker = 'o', alpha = 0.7, linewidths = 1, edgecolors = "k")


    # 画-1到1之间的决策边界
    x1_decision_boundary = np.linspace(-1, 1, 1000)

    x2_decision_boundary = (-logistic_regression_model_1.coef_[0][0] * x1_decision_boundary - logistic_regression_model_1.intercept_[0]) / logistic_regression_model_1.coef_[0][1]


    plt.plot(x1_decision_boundary, x2_decision_boundary,
             label='decision_boundary',
             c='y')

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.legend()

    # (b)
    # 循环获取不同范围C的模型
    plt.figure(figsize=(9, 16))
    line_score_box = []
    for i in range(9): #np.arange(0.001, 10, 1):
        # 实例化支持向量机模型
        C = 0.001*10**i
        svclassifier = SVC(kernel='linear', C=C)
        # 训练线性支持向量机
        svclassifier = svclassifier.fit(x_all_feature, y_all_tag)

        # 使用训完成的线性支持向量机预测数据
        data_linear_predicted = pd.concat([x_all_feature, pd.DataFrame(svclassifier.predict(x_all_feature))], axis=1)

        # 绘制线性支持向量机预测结果的散点图
        plt.subplot(3, 3, i+1)
        data_linear_predicted.columns = [0, 1, 2]
        # compare_in_map(plt=plt, com_pre=data_linear_predicted)
        com_pre = data_linear_predicted
        feature_number = 2
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.title("C=" + str(C))
        plt.scatter(com_pre.loc[com_pre[feature_number] > 0, 0], com_pre.loc[com_pre[feature_number] > 0, 1],
                    label='+1_linear SVC',
                    s=10, c='b', marker='+', alpha=1, linewidths=1, edgecolors='r')
        plt.scatter(com_pre.loc[com_pre[feature_number] < 0, 0], com_pre.loc[com_pre[feature_number] < 0, 1],
                    label='-1_Linear SVC',
                    s=10, c='c', marker='o', alpha=0.9, linewidths=1, edgecolors=None)

        draw_linear_dec(svclassifier)
        print('##########')
        print('C:' + str(C))
        print('coef:' + str(svclassifier.coef_))
        print('intercept:' + str(svclassifier.intercept_))
        print('##########')
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.legend()

    # (c)(i) 四个特征，逻辑回归分类器，模型，训练参数
    data_c = pd.concat([data.iloc[:, 0:2], data[0]**2, data[1]**2, data[2]], axis=1)
    data_c.columns = [0, 1, 2, 3, 4]

    # 分离特征与标签
    x_all_feature_c = data_c.iloc[:, 0:4]
    y_all_tag_c = data_c.iloc[:, 4]

    # 实例化逻辑回归模型
    logistic_regression_model_c = LR(C=1, max_iter=1000)
    # 训练模型
    logistic_regression_model_c = logistic_regression_model_c.fit(x_all_feature_c, y_all_tag_c)
    # 输出模型参数
    print('##### 输出模型参数 #####')
    print("coef_(c):" + str(logistic_regression_model_c.coef_))
    print("intercept_(c):" + str(logistic_regression_model_c.intercept_))
    print('##### #####')

    # (c)(ii)
    # 逻辑回归模型的预测表现
    # logistic_regression_model_1 = logistic_regression_model_1.fit(x_train_feature, y_train_tag)
    # Z = clf.decision_function(xy).reshape(axisx.shape)
    y_tag_predited_c = pd.DataFrame(logistic_regression_model_c.predict(x_all_feature_c))
    x_t_c = x_all_feature_c.reset_index(drop=True)
    com_pre = pd.concat([x_t_c, y_tag_predited_c],axis=1)
    com_pre.columns = [0, 1, 2, 3, 4]

    # 画逻辑回归预测散点图
    plt.figure(figsize=(9, 16))
    # plt.subplot(2, 2, 1)
    print('after:')
    print(com_pre)
    # compare_in_map(4, plt=plt, com_pre=com_pre)

    feature_number = 4
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.scatter(com_pre.loc[com_pre[feature_number] > 0, 0], com_pre.loc[com_pre[feature_number] > 0, 1],
                label='+1_logistic regression(Square data)',
                s=30, c='b', marker='+', alpha=0.9, linewidths=1, edgecolors='m')
    plt.scatter(com_pre.loc[com_pre[feature_number] < 0, 0], com_pre.loc[com_pre[feature_number] < 0, 1],
                label='-1_logistic regression(Square data)',
                s=30, c='c', marker='o', alpha=0.9, linewidths=1, edgecolors='k')




    # 画原始数据和平方数据对比
    # plt.subplot(2, 2, 1)
    # compare_in_map(plt=plt, com_pre=data)

    feature_number = 2
    com_pre=data
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.scatter(com_pre.loc[com_pre[feature_number] > 0, 0], com_pre.loc[com_pre[feature_number] > 0, 1],
                label='+1_original',
                s=10, c='m', marker='+', alpha=1, linewidths=1, edgecolors=None)
    plt.scatter(com_pre.loc[com_pre[feature_number] < 0, 0], com_pre.loc[com_pre[feature_number] < 0, 1],
                label='-1_original',
                s=10, c='k', marker='o', alpha=0.9, linewidths=1, edgecolors=None)

    # **计算决策边界上的点**
    # x1_decision_boundary_c = np.linspace(-1, 1, 1000)
    # logistic_regression_model_2.coef_
    # x2_decision_boundary_c = (-logistic_regression_model_c.coef_[0][0] * x1_decision_boundary_c - logistic_regression_model_c.intercept_[0]) / logistic_regression_model_c.coef_[0][1]

    # (c)(iii) BaseLine model 对比
    # plt.subplot(2,2,2)
    more_feature = 1 if data[2].value_counts().loc[1]>500 else -1
    data_baseline = pandas.DataFrame([[more_feature]]*999)
    x_t_baseline = x_all_feature.reset_index(drop=True)
    com_pre = pd.concat([x_t_baseline, data_baseline],axis=1)
    com_pre.columns = [0, 1, 2]
    # compare_in_map(plt=plt, com_pre=com_pre)

    feature_number = 2
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.scatter(com_pre.loc[com_pre[feature_number] > 0, 0], com_pre.loc[com_pre[feature_number] > 0, 1],
                label='+1_baseLine',
                s=10, c='#bcbd22', marker='+', alpha=1, linewidths=1, edgecolors=None)
    plt.scatter(com_pre.loc[com_pre[feature_number] < 0, 0], com_pre.loc[com_pre[feature_number] < 0, 1],
                label='-1_baseLine',
                s=10, c='#ff7f0e', marker="o", alpha=0.9, linewidths=1, edgecolors=None)
    # print(com_pre)

    plt.rcParams['figure.constrained_layout.use'] = True
    plt.legend()

    plt.show()

def compare_in_map(feature_number=2,
                   s1 = 5, c1 = '#e377c2', marker1 = 'o', alpha1 = 1, linewidths1 = 0, edgecolors1 = None,
                   s2 = 5, c2 = 'b', marker2 = '*', alpha2 = 1, linewidths2 = 0, edgecolors2 = None,
                   xlabel = 'x_1', ylable = 'x_2', legend=None,
                   **kwargs):

    if legend is None:
        legend = ['+1', '-1']
    plt = kwargs['plt']
    com_pre = kwargs['com_pre']

    plt.xlabel(xlabel)
    plt.ylabel(ylable)
    plt.legend(legend, ncol=2)
    # plt.rcParams['figure.constrained_layout.use'] = True

    plt.scatter(com_pre.loc[com_pre[feature_number]>0, 0], com_pre.loc[com_pre[feature_number]>0, 1], label='ss', s = s1, c = c1, marker = marker1, alpha = alpha1, linewidths = linewidths1, edgecolors = edgecolors1) #cmap="rainbow",
    plt.scatter(com_pre.loc[com_pre[feature_number]<0, 0], com_pre.loc[com_pre[feature_number]<0, 1], s = s2, c = c2, marker = marker2, alpha = alpha2, linewidths = linewidths2, edgecolors = edgecolors2)

def draw_linear_dec(model, ax_linear=None):
    if ax_linear is None:
        ax_linear = plt.gca()

    ylim = ax_linear.get_ylim()
    xlim = ax_linear.get_xlim()
    y = np.linspace(ylim[0], ylim[1], 22)
    x = np.linspace(xlim[0], xlim[1], 22)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    ax_linear.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    ax_linear.set_xlim(xlim)
    ax_linear.set_ylim(ylim)

if __name__ == "__main__":
    main()


    # #################################
    # from sklearn.linear_model import LinearRegression
    # model = LinearRegression().fit(Xtrain, ytrain)
    # print(model.intercept_, model.coef_)
    # ypred = model.predict(Xtrain)
    # import matplotlib.pyplot as plt
    # plt.rc('font', size=18)
    # plt.rcParams['figure.constrained_layout.use'] = True
    # plt.scatter(Xtain, ytrain, color='black')
    # plt.plot(Xtain, ypred, color='blue', linewidth=3)
    # plt.xlabel('input x');
    # plt.ylable("outpout y')
    # plt.legend(['predictions', 'training data'])
    # plt.show()

    # import numpy as np
    # Xtrain_poly = np.column_stack((Xtrain, Xtrain ** 2))
    # model = LinearRegression().fit(Xtrain_poly, ytrain)
    # print(model.intercept_, model.coef_)
    # ypred = model.predict(Xtrain_poly)
    # plt.scatter(Xtrain, ytrain, color='black')
    # plt.plot(Xtrain, ypred, color='blue', linewidth=3)
    # plt.xlabel('input x');
    # plt.ylablel('output y')
    # plt.legend(['predictions', 'training data'])
    # plt.show()

    #################################################

    # print('#################ypp')
    # print(s1)
    # c1 = kwargs['c1'] or "b"
    # marker1 = kwargs['marker1'] or "o"
    # alpha1 = kwargs['alpha1'] or 1
    # linewidths1 = kwargs['linewidths1'] or 0
    # edgecolors1 = kwargs['edgecolors1'] or None

    # s2 = kwargs['s2'] or 5
    # c2  = kwargs['c2'] or "b"
    # marker2 = kwargs['marker2'] or "x"
    # alpha2 = kwargs['alpha2'] or 1
    # linewidths2 = kwargs['linewidths2'] or 0
    # edgecolors2 = kwargs['edgecolors2'] or None

    # s1 = 5, c1 = 'b', marker1 = 'o', alpha1 = 1, linewidths1 = 0, edgecolors1 = None,
    # s2 = 5, c2 = 'r', marker2 = 'x', alpha2 = 1, linewidths2 = 0, edgecolors2 = None)

    # s = "s1",
    # c = "c1",
    # marker = "marker1",
    # alpha = "alpha1",
    # linewidths = "linewidths1",
    # edgecolors = "edgecolors1"
