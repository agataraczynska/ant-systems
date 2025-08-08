import numpy as np
import matplotlib.pyplot as plt
import math
import random

def euclidean_distance(x,y):
    d = np.zeros((len(x),len(y)), dtype=float, order='C')
    for i in range(len(x)):
        for j in range(len(y)):
            d[i,j] = math.sqrt( (x[i]-x[j])**2 + (y[i]-y[j])**2 )
    return d

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def pR(U, L, k, d):
    p_r = ((U + k) ** d) / ((U + k) ** d + (L + k) ** d)
    return p_r

def A(sp, sd, alpha=1, beta=5):
    a = (sp ** alpha * (1 / sd) ** beta) / (sum(sp ** alpha * (1 / sd) ** beta))
    return a


def task1_ant_bridge():
    print("Nest - food environment\n")

    x = [0,1,2,2,3,4,5,5,6,7]
    y = [2,2,0,3,2,2,4,1,2,2]

    # Plot map
    x1 = [0,1,2,3,4,5,6,7]
    y1 = [2,2,0,2,2,4,2,2]
    x2 = [0,1,2,3,4,5,6,7]
    y2 = [2,2,3,2,2,1,2,2]

    fig, ax = plt.subplots()
    ax.plot(x1, y1,'co-', linewidth=0.5, markersize=5)
    ax.plot(x2, y2,'co-', linewidth=0.5, markersize=5)
    l = ['Nest','','','','','','','','','Food']
    for i, txt in enumerate(l):
        ax.annotate(txt, (x[i], y[i]), fontsize=13)
    plt.show()

    num_ants = 1000
    Upper_bridge = [0,0]
    Lower_bridge = [0,0]
    k = 20
    d = 2

    distances = euclidean_distance(x,y)
    unique_moves = unique(x)

    for i in range(num_ants):
        for j in range(len(unique_moves)):
            r = random.uniform(0, 1)
            if j == 2:
                U = Upper_bridge[0]
                L = Lower_bridge[0]
                p_R = pR(U,L,k,d)
                if r <= p_R:
                    Upper_bridge[0] += 1
                else:
                    Lower_bridge[0] += 1
            elif j == 5:
                U = Upper_bridge[1]
                L = Lower_bridge[1]
                p_R = pR(U,L,k,d)
                if r <= p_R:
                    Upper_bridge[1] += 1
                else:
                    Lower_bridge[1] += 1

    fig, ax = plt.subplots()
    ax.plot(x1, y1,'co-', linewidth=0.5, markersize=5)
    ax.plot(x2, y2,'co-', linewidth=0.5, markersize=5)
    l = ['Nest','',Upper_bridge[0],Lower_bridge[0],'','',Lower_bridge[1],Upper_bridge[1],'Food']
    for i, txt in enumerate(l):
        ax.annotate(txt, (x[i], y[i]), fontsize=13)
    plt.show()

def task2_tsp_ant():
    print("\nTravelling salesman problem\n")

    x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
    y = [1, 4, 5, 3, 0,  4,  10, 6, 9, 10]

    Tmax = 200
    alpha = 1
    beta = 5
    ro = 0.5
    ants = 10
    T = 0
    N = 10

    start = 0
    best = []
    best_ph = []
    pheromones = np.ones((len(x), len(y)))
    distances = euclidean_distance(x,y)

    while T < Tmax:
        ph_canals = np.zeros(distances.shape + (ants,))
        routes = []

        for ant in range(ants):
            unvisited_cities = list(range(len(x)))
            proposed_route = []
            proposed_route.append(start)
            current_city = unvisited_cities.pop(start)
            quantity = np.zeros(distances.shape)

            for i in range(N-1):
                teta = pheromones[current_city, unvisited_cities]
                eta = distances[current_city, unvisited_cities]
                a = A(teta, eta, alpha, beta)
                p = a / sum(a)
                next_city = np.random.choice(unvisited_cities, p=p)
                unvisited_cities.remove(next_city)
                quantity[current_city, next_city] = 1 / distances[current_city, next_city]
                current_city = next_city
                proposed_route.append(next_city)

            quantity[current_city, start] = 1 / distances[current_city, start]
            proposed_route.append(start)
            ph_canals[:, :, ant] = quantity
            routes.append(proposed_route)

        ph_sum = np.sum(np.sum(ph_canals, axis=1), axis=0)
        index_best = np.argmax(ph_sum)
        best.append(routes[index_best])
        best_ph.append(ph_sum[index_best])
        delta_ph = np.sum(ph_canals, axis=2)
        pheromones = (1-ro) * pheromones + delta_ph

        T += 1

    np.savetxt('pheromones.csv', pheromones, delimiter=',', fmt='%s')

    best_idx = best_ph.index(max(best_ph))
    best_route = best[best_idx]
    print("Travelling salesman problem - best route:", best_route)
    dist = 0
    for i in range(1, len(best_route)):
        c = best_route[i-1]
        n = best_route[i]
        dist += distances[c,n]
    print("Travelling salesman problem - total distance traveled:", dist)

    x2 = []
    y2 = []
    for i in range(N):
        x2.append(x[best_route[i]])
        y2.append(y[best_route[i]])
    x2.append(x[best_route[0]])
    y2.append(y[best_route[0]])

    l = list(range(N))
    fig, ax = plt.subplots()
    ax.plot(x2[0:N+1], y2[0:N+1], 'co-', linewidth=0.5, markersize=5)
    for i, txt in enumerate(l):
        ax.annotate(txt, (x[i], y[i]), fontsize=13)
    plt.show()


if __name__ == "__main__":
    task1_ant_bridge()
    task2_tsp_ant()
