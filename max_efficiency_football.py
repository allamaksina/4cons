"""Для поиска максимального значения целевой функции используется
генетический алгоритм с добавлением шума, как вариант направленного перебора"""

import pandas as pd
import numpy as np

df = pd.read_csv('football.csv')
df.drop('id', axis=1)

df['goals_rate'] = df.Goals / df.games
# df.sort_values(['goals_rate', 'Goals'], ascending=False, inplace=True)
np.random.seed(42)
df['price'] = np.random.randint(10000, 1000000, (len(df), 1))
df['price'] = df['price'] * df['goals_rate']
df.head()

BUDGET = 5_000_000
N_PLAYERS = 11

POPULATION = 300
MAX_PARENTS = 20
MAX_CHILDREN = 100

N_ITERS = 5000


def objective_fun(x: np.array, a: np.array) -> float:
    return a @ x


def budget_restriction(price: np.array, a: np.array, max_value) -> float:
    return price @ a <= max_value


def generate_children(ancestors, n_ch, big_length, length, prob):
    for _ in range(n_ch):
        child = np.zeros((big_length,))
        rand_indices = np.random.choice(len(ancestors), replace=False, size=(2,))
        ans = np.concatenate((np.where(ancestors[rand_indices[0]][0] > 0)[0],
                             np.where(ancestors[rand_indices[1]][0] > 0)[0]),
                             axis=0)

        for _ in range(length):
            if np.random.uniform(0, 1) < prob:
                child[np.random.randint(0, big_length)] = 1
            else:
                child[np.random.choice(ans)] = 1
        ancestors.append([child, 0])

    return ancestors


def main():

    x = np.array(df['goals_rate'])
    price = np.array(df['price'])
    mutation_prob = 0.9

    # initial generation of a
    parents = []
    for _ in range(POPULATION):
        a = np.zeros((len(df),))
        counter = 0
        while counter < N_PLAYERS:
            i = np.random.randint(len(df))
            if not a[i]:
                counter += 1
                a[i] = 1
        parents.append([a, 0])

    epoch = 0

    while epoch < N_ITERS:

        valid_parents = []

        for p in parents:
            if budget_restriction(price, p[0], BUDGET):
                p[1] = objective_fun(x, p[0])
                valid_parents.append(p)

        parents = valid_parents.copy()
        valid_parents.clear()

        parents.sort(key=lambda z: z[1], reverse=True)

        parents = parents[:MAX_PARENTS]

        parents = generate_children(parents,
                                    MAX_CHILDREN,
                                    len(df),
                                    N_PLAYERS,
                                    mutation_prob)

        if epoch % 100 == 0:
            mutation_prob *= 0.99
            print(f'epoch: {epoch:5d},  sum: {parents[0][1]:03.1f}, '
                  f'mutation probability {mutation_prob:.3f}')
        epoch += 1


if __name__ == "__main__":
    main()
