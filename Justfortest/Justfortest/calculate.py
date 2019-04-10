# coding:utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import random
import math
import operator
from math import sin, asin, cos, radians, fabs, sqrt
from itertools import combinations
import time
from io import BytesIO

def haversine(lat_0, lon_0, lat_1, lon_1):
    """
    This function is to do with some calculations about the longitude and
    latitude, given 2 points with longitude and latitude, return the distance
    between the 2 points.

    Args:
        lat_0, lon_0: The latitude and longitude of one point on the map
        lat_1, lon_1: The latitude and longitude of another point on the map

    Returns:
        distance: The distance between 2 points (km)

    """
    # 经纬度转换成弧度

    lat_0 = radians(lat_0)
    lat_1 = radians(lat_1)
    lon_0 = radians(lon_0)
    lon_1 = radians(lon_1)
    d_lon = fabs(lon_0 - lon_1)
    d_lat = fabs(lat_0 - lat_1)
    h = sin(d_lat / 2) ** 2 + cos(lat_0) * cos(lat_1) * sin(d_lon / 2) ** 2
    EARTH_RADIUS = 6371
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))

    return distance


# def init_para():
# 	alpha = 0.96
# 	t = (1,500)
# 	markovlen = 500

# 	return alpha,t,markovlen


def init_para():
    """

    Initialize the parameters.

    """
    alpha = lala
    t = (1, lalala)
    markovlen = lalala

    return alpha, t, markovlen


def get_distmat(coordinates, num):
    """
    This function is to transform the location of every point into every distance
    between 2 points

    Args:
        coordinates: The Two-dimensional arraylist which shows every point's
            location on the map in longitude and latitude.
        num: The num of the points on the map

    Returns:
        distmat: A Two-dimensional arraylist which includes every 2 point's distance
            to each other (km)

    """
    # num = coordinates.shape[0] #坐标点
    distmat = np.zeros((num, num))  # 距离矩阵
    for i in range(num):
        for j in range(i, num):
            # distmat[i][j] = distmat[j][i]=np.linalg.norm(coordinates[i]-coordinates[j])
            distmat[i][j] = distmat[j][i] = 1000 * haversine(coordinates[i][1], coordinates[i][0], coordinates[j][1],
                                                             coordinates[j][0], )
    return distmat


def rep(list1, list2):
    """
    This function is arecursive function, to full permutate the generated plan in
    order to prevent from missing solutions

    Args:
        list1, list2 : Two list

    Returns:
        A new list which has been permutated

    """

    copylist = list2[0:]

    last = list1.pop()
    if list2 == []:
        list2.append([last])
    else:
        for i in range(len(copylist)):
            list2.pop()
            for j in range(len(copylist[i]) + 0):  # 本来全排列需要加一，但多重循环会严重拖慢代码速度
                templist = copylist[i][0:]
                templist.insert(j, last)
                list2.insert(0, templist)

    list2 = list(set([tuple(i) for i in list2]))
    list2 = [list(v) for v in list2]
    if list1 == []:
        return list2

    else:
        return rep(list1, list2)


def PSO(distmat, solution):
    # initlization()
    N = 50  # 种群规模
    tmax = 1000  # 迭代次数
    # exlist=[]         #速度,也即是交换序列
    # particle=[]           #位置
    # Pbest=                #个体最佳
    # Gbest=                #群体最佳
    w = 0.729  # 惯性权重
    c1 = c2 = 1.49445

    particle = []
    num = len(solution)

    for i in range(N):
        # lala=np.arange(num)
        np.random.shuffle(solution)
        jaja = solution.tolist()
        particle.insert(i, jaja)
    # print particle[i].dtype
    # print len(particle)
    # print particle
    # print particle[29]
    Gbest = 99999999
    Pbest = []
    exlist = []
    fit = np.arange(N).tolist()
    Psoulution = []
    for i in range(N):
        Pbest.append(99999999)
        exlist.insert(i, [])
        Psoulution.insert(i, [])
        for j in range(num):
            a = np.random.randint(num)
            b = np.random.randint(num)
            exlist[i].insert(j, [a, b])

    result = []
    Bestever = 0
    tmplist = []
    for t in range(tmax):

        for i in range(N):
            # print i
            # print particle[i]
            # destabilization(particle[i],num)
            # fit[i]=fitness(particle[i])

            # n=len(list)
            fit[i] = 0
            for j in range(len(particle[i]) - 2):
                fit[i] += distmat[particle[i][j]][particle[i][j + 1]]
            fit[i] += distmat[0][particle[i][0]] + distmat[0][particle[i][len(particle[i]) - 1]]

            if fit[i] < Pbest[i]:
                Pbest[i] = fit[i]
                Psoulution[i] = particle[i][:]
            else:
                if np.random.rand() < np.exp(-(fit[i]-Pbest[i])*i/tmax):
                    Pbest[i]=fit[i]
                    Psoulution[i]=particle[i][:]
            if fit[i] < Gbest:
                Gbest = fit[i]
                Gsoulution = particle[i][:]
            # Bestever=i

            tmplist = []
            tmplist1 = []
            tmplist2 = []
            # tmplist1=temp(Psoulution[i],particle[i],c1)
            # tmplist2=temp(Gsoulution,particle[i],c2)
            for k in range(num):
                jaja = (k, Psoulution[i].index(particle[i][k]), c1)
                tmplist1.append(jaja)
            for k in range(num):
                jaja = (k, Gsoulution.index(particle[i][k]), c2)
                tmplist2.append(jaja)
            # if particle[i] != Gbest:

            tmplist = tmplist1 + tmplist2

            # reset(particle[i],tmplist)
            for j in range(len(tmplist)):
                jaja = tmplist[j]
                if np.random.random() <= jaja[2]:
                    particle[i][jaja[0]], particle[i][jaja[1]] = particle[i][jaja[1]], particle[i][jaja[0]]

        result.append(Gbest)
    # print("hahahha",type(Gsoulution))

    return [Gsoulution, result]


def SA(distmat, solutionnew):
    """

    Simulate Anneal.
    This is the most important functionin the whole program.
    The method models the physical process of heating a material and then slowly
    lowering the temperature to decrease defects, thus minimizing the system energy.

    At each iteration of the simulated annealing algorithm, a new point is randomly
    generated. The distance of the new point from the current point, or the extent of
    the search, is based on a probability distribution with a scale proportional to the
    temperature. The algorithm accepts all new points that lower the objective, but also,
    with a certain probability, points that raise the objective. By accepting points that
    raise the objective, the algorithm avoids being trapped in local minima, and is able
    to explore globally for more possible solutions. An annealing schedule is selected to
     systematically decrease the temperature as the algorithm proceeds. As the temperature
     decreases, the algorithm reduces the extent of its search to converge to a minimum.

    Args:
        distmat: A Two-dimensional arraylist which includes every 2 point's distance to each other (km)
        solutionnew: A Initial solution in ndarray type

    Returns:
        solutionbest: List. The best ( maybe it's not the best, but the closest one) Point
            sequence which minimize the cost to traverse all points .

        result: A list which contains the current solution's cost in each iteration of
            Simulate Anneal, which shows the process of the convergence of the solution.

    """

    # solutionnew = np.arange(num)

    # valuenew = np.max(num)
    num = len(solutionnew)
    solutioncurrent = solutionnew.copy()
    valuecurrent = 9999999000  # np.max这样的源代码可能同样是因为版本问题被当做函数不能正确使用，应取一个较大值作为初始值
    # print(valuecurrent)

    solutionbest = solutionnew.copy()
    global valuebest
    valuebest = 9999999000  # np.max

    alpha, t2, markovlen = init_para()
    t = t2[1]

    result = []  # 记录迭代过程中的最优解
    while t > t2[0]:
        for i in np.arange(markovlen):

            # 下面的两交换和三角换是两种扰动方式，用于产生新解
            if np.random.rand() > 0.5:  # 交换路径中的这2个节点的顺序
                # np.random.rand()产生[0, 1)区间的均匀随机数
                while True:  # 产生两个不同的随机数
                    loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    # print(loc1,loc2)
                    # print(result)
                    if loc1 != loc2:
                        break
                solutionnew[loc1], solutionnew[loc2] = solutionnew[loc2], solutionnew[loc1]
            else:  # 三交换
                while True:
                    loc1 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc2 = np.int(np.ceil(np.random.rand() * (num - 1)))
                    loc3 = np.int(np.ceil(np.random.rand() * (num - 1)))

                    if ((loc1 != loc2) & (loc2 != loc3) & (loc1 != loc3)):
                        break

                    # 下面的三个判断语句使得loc1<loc2<loc3
                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1
                if loc2 > loc3:
                    loc2, loc3 = loc3, loc2
                if loc1 > loc2:
                    loc1, loc2 = loc2, loc1

                # 下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
                tmplist = solutionnew[loc1:loc2].copy()
                solutionnew[loc1:loc3 - loc2 + 1 + loc1] = solutionnew[loc2:loc3 + 1].copy()
                solutionnew[loc3 - loc2 + 1 + loc1:loc3 + 1] = tmplist.copy()

            valuenew = 0
            for i in range(num - 1):
                valuenew += distmat[solutionnew[i]][solutionnew[i + 1]]
            valuenew += distmat[solutionnew[0]][solutionnew[num - 1]]
            # print (valuenew)
            if valuenew < valuecurrent:  # 接受该解

                # 更新solutioncurrent 和solutionbest
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()

                if valuenew < valuebest:
                    valuebest = valuenew
                    solutionbest = solutionnew.copy()
            else:  # 按一定的概率接受该解
                if np.random.rand() < np.exp(-(valuenew - valuecurrent) / t):
                    valuecurrent = valuenew
                    solutioncurrent = solutionnew.copy()
                else:
                    solutionnew = solutioncurrent.copy()
        t = alpha * t
        result.append(valuebest)
        print(t)  # 程序运行时间较 长，打印t来监视程序进展速度
    # 用来显示结果
    print("SBEST:", solutionbest)
    print("VBEST:", valuebest)
    return [solutionbest, result]


def check_one_way(newway, list1, list2):
    """
    Check whether the assignment of a single uav meets the constraints

    Args:
        list1: The task assignment list
        list2: The constraint list
        newway: The solution list

    Returns:
        bool: Whether the assignment of a single uav meets the constraints

    """
    n = len(list2)
    bool = 1
    for i in range(n):
        long = check_length(newway[list1[i]:list1[i + 1]])
        if long > list2[i]:
            bool = 0
        else:
            continue

    return bool


def check_way(newway, list1, list2):
    """
    Check whether the assignments of all the uav meets the constraints

    Args:
        list1: The task assignment list
        list2: The constraint list
        newway: The solution list

    Returns:
        bool: Whether the assignments of all the uav meets the constraints

    """

    bool = 0
    for i in range(len(list2)):
        bool += check_one_way(newway, list1, list2[i])
    return bool


def check_length(list):
    """
    Caculate the length of the path

    Args:
        list: The list of points sequence in the path

    Returns:
        length: The length of the path

    """

    n = len(list)
    length = 0
    for i in range(n - 2):
        length += distmat[list[i]][list[i + 1]]
    length += distmat[0][list[0]] + distmat[0][list[n - 1]]
    return length


def extra(list, soul):
    """
    Caculate the extra cost for an assignment.

    Args:
        list: The assignment which needs to be verified
        soul: Annealed solution which minimize the cost to traverse all points .

    Returns:
        extra: The extra cost of the path in this assignment. Smaller means better.

    """

    j = len(list)
    jj = float(j)
    extra = 0
    if j == 1:
        extra = 1000000
    else:
        for i in range(1, j - 1):
            fitness = distmat[soul[list[i]]][0] + distmat[soul[list[i] - 1]][0] - distmat[soul[list[i] - 1]][
                soul[list[i]]]
            extra += fitness
    # 此处防止出现过短的方案导致分配不均衡
    return extra


def caculation(answer):
    """
    Caculate the path length for the generated assignment

    Args:
        answer: The generated assignment

    Returns:
        mile: The length of the whole path of the generated assignment

    """

    s = len(answer)
    mile = 0
    for i in range(s):
        mile += check_length(answer[i])
    return mile


def total_weight(answer):
    """
    Caculate the total weight for the generated assignment

    Args:
        answer: The generated assignment

    Returns:
        t_weight: The total weight of the whole path of the generated assignment

    """

    s = len(answer)
    t_weight = 0
    for i in range(s):
        jj = len(answer[i])
        for j in range(jj):
            t_weight += weight[answer[i][j]]
    return t_weight


def recut(soul, answer, require):
    """
    This function is a circle process to continuously truncate those points with low fitness.

    Args:
        soul: Annealed solution which minimize the cost to traverse all points .
        answer: The generated assignment.
        require: The range list for every uav.

    Returns:
        [soul,answer,ready]
        soul: The same one
        answer: The answer list which has been cut.
        ready: Whether the assignments of the answer meets the constraints

    """

    voyage = 0
    for i in range(len(require)):
        voyage += require[i]
    mile = caculation(answer)
    # points=int((1-voyage/mile)*len(soul)+1)
    points = int(num / 10)
    print("points", points)
    # list=radius(soul)
    # list=distancechange(soul)
    # list=distanceandweightchange(soul,answer)
    haha = len(soul)
    dic = {}
    output = []
    dic[soul[1]] = 0

    mile = caculation(answer)
    wwweight = total_weight(answer)

    meandistant = valuebest / num
    for i in range(1, haha - 1):
        deltadistance = distmat[soul[i]][soul[i - 1]] + distmat[soul[i]][soul[i + 1]] - distmat[soul[i - 1]][
            soul[i + 1]]  # 删除增益
        divtans = deltadistance / meandistant
        deltaweight = weight[soul[i]]
        lalal = mile / wwweight
        jajaj = (mile - deltadistance) / (wwweight - deltaweight)

        revalue = divtans * divtans / weight[soul[i]]  # 评估函数该怎么处理
        # revalue=lalal-jajaj
        # revalue=jajaj-lalal
        dic[soul[i]] = revalue
    dic[soul[haha - 1]] = 0
    sortdic = sorted(dic.items(), key=operator.itemgetter(1))
    for i in range(haha - 1):
        output.insert(i, sortdic[i][0])
    list = output
    print("list", list)
    print("soul", soul)
    # cutlist=list[-points:]
    # haha=len(cutlist)
    for i in range(points):
        # del soul[cutlist[i]]
        # del soul[list[len(soul)-i]]
        soul.remove(list[len(list) - i - 1])
    answer = designate(soul, constraint)[0]
    ready = designate(soul, constraint)[1]
    return [soul, answer, ready]


# 负责具体的分派任务部分，给定解序列，和一个航程约束数组，返回一个answer，
# 包含每个无人机各自所要遍历的点，也即具体任务分派
def designate(soul, require):
    """
    This function is the main part about the assignmet of the uav.
    Given an annealed solution and a require list of every uav's range.
    Then output the answer, which includes the Specific task assignment
    for each uav.

    Args:
        soul: Annealed solution which minimize the cost to traverse all points .
        require: The range list for every uav.

    Returns:
        [THEANSWER,bool]
        THEANSWER: The current answer to this problem, which is a list contains
            every uav's task, every task contains all the points for the
            corresponding uav to traverse.
         bool: Shows whether this answer meets the constraint, if not, then cycle
            again, else, the current anwer will be the final solution.


    """

    k = len(require)

    num = len(soul) + 1
    hahah = [i for i in range(1, num - 1)]
    combine = list(combinations(hahah, k - 1))
    for i in range(len(combine)):
        # for j in range(0):
        newcombine = list(combine[i])
        # newcombine=[x+j for x in list(combine[i])]
        newcombine.append(num - 1)
        newcombine.insert(0, 0)
        combine[i] = newcombine
    # tasks=alltheroads(k-1,soul)
    tasks = combine

    # print tasks
    s = len(tasks)
    methondcurrent = tasks[0]
    for i in range(s):
        if extra(tasks[i], soul) > extra(methondcurrent, soul):
            methondcurrent = tasks[i]
    mworst = methondcurrent

    for i in range(s):
        methondnew = tasks[i]
        ss = constraint[0:]
        aa = rep(ss, [])
        if check_way(soul, tasks[i], aa) != 0:
            if extra(methondnew, soul) < extra(methondcurrent, soul):
                methondcurrent = methondnew

            else:
                continue

    mbest = methondcurrent
    bool = 1
    if mbest == mworst:
        bool = 0
    print("MBEST", mbest)
    print("EB", extra(mbest, soul))
    THEANSWER = []
    for i in range(k):
        THEANSWER += [soul[mbest[i]:mbest[i + 1]]]
    if len(require) == 1:
        ss = constraint[0:]
        bool = check_one_way(soul, tasks[0], ss)
        THEANSWER = [soul[0:]]

    return [THEANSWER, bool]


# # 画图部分
# def paint(soul):

# 	"""

# 	Show all the path on the map in a picture.

# 	"""

# 	v=len(soul)
# 	colors=['r','c','g','m','y','b','k','r','c','g','m','y','b','k']


# 	plt.figure()
# 	left,bottom,width,height=0,0,0.7,1
# 	plt.axes([bottom,left,width,height])
# 	plt.title("The Map")
# 	plt.rcParams['xtick.direction']='in'
# 	plt.axis([31.10,31.11,121.28,121.29])
# 	plt.grid(True)
# 	plt.scatter(coordinates[0][0],coordinates[0][1],color='yellow',s=150)
# 	for i in range(0,num-1):
# 		plt.scatter(coordinates[i][0],coordinates[i][1],color='b')
# 		plt.annotate(i,(coordinates[i][0],coordinates[i][1]))
# 	for i in range(v):
# 		r=len(soul[i])
# 		kk=soul[i][1]
# 		ll=soul[i][0]
# 		a=[coordinates[ll][0],coordinates[kk][0]]
# 		b=[coordinates[ll][1],coordinates[kk][1]]
# 		plt.plot(a,b,color=colors[i],label=('The route of uav',i+1))
# 		for j in range(r-1):
# 			k=soul[i][j+1]
# 			l=soul[i][j]
# 			m=[coordinates[l][0],coordinates[k][0]]
# 			n=[coordinates[l][1],coordinates[k][1]]
# 			plt.plot(m,n,color=colors[i])


# 	#left,bottom,width,height=0.2,0.8,0.2,0.2
# 	#plt.legend(loc=SouthEast)
# 	#plt.legend(bbox_to_anchor=(1,1))
# 	left,bottom,width,height=0.3,0.75,0.25,0.7
# 	plt.axes([bottom,left,width,height])
# 	plt.title("The result")
# 	plt.rcParams['xtick.direction']='in'
# 	plt.plot(np.array(result))  
# 	plt.ylabel("bestvalue")  
# 	plt.xlabel("t")
# 	left,bottom,width,height=0,0.7,0.3,0.3
# 	plt.axes([bottom,left,width,height])
# 	plt.axis('off')		
# 	jaja=[0,0]
# 	yaya=[0,0]
# 	for i in range(v):
# 		plt.plot(jaja,yaya,color=colors[i],label=('The route of uav',i+1))
# 	plt.legend(loc='center')
# 	plt.show()


def paint(soul):
    """

    Show all the path on the map in a picture.

    """
    v = len(soul)
    colors = ['r', 'c', 'g', 'm', 'y', 'b', 'k', 'r', 'c', 'g', 'm', 'y', 'b', 'k']

    plt.figure(1)
    # left,bottom,width,height=0,0,0.7,1
    # plt.axes([bottom,left,width,height])
    plt.title("The Map")
    plt.rcParams['xtick.direction'] = 'in'
    #plt.axis([15.150,15.175,37.600,37.630])
    # plt.axis([103.4125, 103.6600, 30.9300, 31.1580])
    plt.grid(True)
    plt.scatter(coordinates[0][0], coordinates[0][1], color='yellow', s=150)
    # plt.scatter(coordinates[num-1][0],coordinates[num-1][1],color='b')
    for i in range(0, num):
        plt.scatter(coordinates[i][0], coordinates[i][1], color='b')
        # plt.annotate(i+1,(coordinates[i][0],coordinates[i][1]))
        plt.annotate(i + 1, (coordinates[i][0], coordinates[i][1]))
    for i in range(v):
        r = len(soul[i])
        kk = soul[i][1]
        ll = soul[i][0]
        a = [coordinates[ll][0], coordinates[kk][0]]
        b = [coordinates[ll][1], coordinates[kk][1]]
        plt.plot(a, b, color=colors[i], label=('The route of uav', i + 1))
        for j in range(r - 1):
            k = soul[i][j + 1]
            l = soul[i][j]
            m = [coordinates[l][0], coordinates[k][0]]
            n = [coordinates[l][1], coordinates[k][1]]
            plt.plot(m, n, color=colors[i])

    # left,bottom,width,height=0.2,0.8,0.2,0.2
    # plt.legend(loc=SouthEast)
    # plt.legend(bbox_to_anchor=(1,1))
    # left,bottom,width,height=0.3,0.75,0.25,0.7

    #left, bottom, width, height = 0, 0.5, 0.1, 0.1
    #plt.axes([bottom, left, width, height])

    jaja= [0, 0]
    yaya= [0, 0]
    # for i in range(v):
    #     plt.plot(jaja, yaya, color=colors[i], label=('The route of uav', i + 1))
    # plt.legend(loc='best')

    buffer = BytesIO()
    plt.savefig(buffer)
    plot_data = buffer.getvalue()
    return plot_data

    # plt.figure(2)
    # # plt.axes([bottom,left,width,height])
    # plt.title("The result")
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.plot(np.array(result))
    # plt.ylabel("bestvalue")
    # plt.xlabel("t")
    # plt.show()

# # 善后事宜
# def FOLLOWUP(newsoul,result):
# 	print ("newsoul",newsoul)
# 	THEANSWER=designate(newsoul,constraint)[0]
# 	READY=designate(newsoul,constraint)[1]
# 	print ("THEANSWER",THEANSWER)
# 	print ("READY",READY)

# 	miles=caculation(THEANSWER)
# 	print(miles)
# 	# ways=check(THEANSWER)

# 	temp_mile = 0
# 	temp_way = []
# 	for i in range(len(THEANSWER)):
# 		temp_mile = check_length(THEANSWER[i])
# 		temp_way += [temp_mile]
# 	ways=temp_way
# 	print(ways)

# 	tempanswer = THEANSWER[:]
# 	for i in range(len(tempanswer)):
# 		tempanswer[i].insert(0,0)
# 		tempanswer[i] += [0]


# 	THEWAY=tempanswer

# 	print(THEWAY)
# 	# evaluate(THEANSWER)
# 	alldistance=caculation(THEANSWER)
# 	allweight=total_weight(THEANSWER)
# 	distweight=allweight/alldistance
# 	print(alldistance)
# 	print(allweight)
# 	print(distweight)
# 	#print(check_way(newsoulwithout,mbest,[4000,4000,4000]))

# 	paintttt(THEWAY)


def main(cons, dataframe, number, alphaa, tt):
    """

    Main function.

    """

    # num=len(dataframe)
    global lala
    lala = alphaa
    global lalala
    lalala = tt
    global num
    num = number
    coor = []
    global weight
    weight = []
    for i in range(num):
        coor.insert(i, [dataframe.iloc[i, 0], dataframe.iloc[i, 1]])
        weight.insert(int(i), dataframe.iloc[i, 2])
    # coor.insert(0, [103.6567,30.9819])
    global coordinates
    coordinates = coor
    global constraint
    constraint = cons
    global distmat
    global alldistance
    distmat = get_distmat(coordinates, num)
    # THISISEND()
    initsoul = np.arange(num)

    # tempsolution = PSO(distmat, initsoul)
    # solutemp = tempsolution[0]
    # resultemp = tempsolution[1]
    # initsoul = np.array(solutemp)

    SSA = SA(distmat, initsoul)

    solutionbest = SSA[0]
    global result

    result=SSA[1]
    # result = resultemp + SSA[1]

    global newsoul
    newsoul = solutionbest.tolist()
    del newsoul[0]

    print("newsoul", newsoul)
    Endup = designate(newsoul, constraint)
    THEANSWER = Endup[0]
    READY = Endup[1]  # 代表这个生成方案是否符合约束条件
    while READY == 0:  # 不符合就进行recut循环，从所有点中舍去一部分
        newsoul = recut(newsoul, THEANSWER, constraint)[0]
        newsoul = np.array(newsoul, dtype=int)
        newsoul = SA(distmat, newsoul)[0].tolist()
        # newsoul.tolist()
        THEANSWER = recut(newsoul, THEANSWER, constraint)[1]
        READY = recut(newsoul, THEANSWER, constraint)[2]
        if len(newsoul) < int(num / 10):
            break
            print("this is a fake uav")  # 无人机航程过小时才会出现这种情况
    # FOLLOWUP(newsoul,result)		# 根据现有的可以生成可行解的点集分配方案并输出图片等信息

   #print("newsoul", newsoul)
    THEANSWER = designate(newsoul, constraint)[0]
    READY = designate(newsoul, constraint)[1]
    #print("THEANSWER", THEANSWER)
   # print("READY", READY)

    miles = caculation(THEANSWER)
    #print(miles)
    # ways=check(THEANSWER)

    temp_mile = 0
    temp_way = []
    for i in range(len(THEANSWER)):
        temp_mile = check_length(THEANSWER[i])
        temp_way += [temp_mile]
    ways = temp_way
   #print(ways)

    tempanswer = THEANSWER[:]
    for i in range(len(tempanswer)):
        tempanswer[i].insert(0, 0)
        tempanswer[i] += [0]

    THEWAY = tempanswer

    print(THEWAY)
    # evaluate(THEANSWER)
    alldistance = caculation(THEANSWER)
    allweight = total_weight(THEANSWER)
    distweight = allweight / alldistance
    print(alldistance)
    # print(check_way(newsoulwithout,mbest,[4000,4000,4000]))

    img = paint(THEWAY)
    return [THEWAY, img]


if __name__ == '__main__':
    # aa=rep([1200,1500,1200,1200],[])
    # cons,dataframe,number,alphaa,tt=pa.para()
    # Target(81Points)
    # temp = pd.read_excel('Target(81Points).xlsx')
    temp = pd.read_excel('Wenchuan(50Points).xlsx')

    cons, dataframe, number, alphaa, tt = [80000,80000,80000], temp, len(temp), 0.97, 1000
    distances = []
    # 分别对应约束数组，数据表噶，点数量，退火系数，初始温度
    # for i in range(10):
    #     main(cons, dataframe, number, alphaa, tt)
    #     distances.append(alldistance)
    # print("----FINAL RESULTS----")
    # for temp in range(len(distances)):
    #     print(distances[temp])
    main(cons, dataframe, number, alphaa, tt)
