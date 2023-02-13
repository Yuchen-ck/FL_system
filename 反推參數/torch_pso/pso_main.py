import random
import numpy as np
from read_model import *

def target_func(x,y):
    ANN_function = torch_model_function(x)
    return ANN_function - y

def fitness(x,y):
    return abs(target_func(x,y))

class Particle:
    def __init__(self, x0):
        self.position = np.array(x0)
        self.velocity = np.array([0, 0 ,0])
        self.best_position = self.position
        self.best_value = float('inf')
        self.value = float('inf')


    def update(self, best_global_position):
        # 計算粒子的速度，根據粒子的最佳位置、全局最佳位置以及隨機數
        self.velocity = 0.7 * self.velocity + 0.5 * np.random.random(3) * (self.best_position - self.position) + 0.5 * np.random.random(3) * (best_global_position - self.position)
        self.position = self.position + self.velocity # 更新粒子的位置
        self.value = fitness(self.position ,y) # 計算粒子的當前解的值
        
        if self.value < self.best_value: # 如果當前解的值比最佳解的值更好
            self.best_value = self.value # 更新粒子的最佳解的值
            self.best_position = self.position # 更新粒子的最佳位置


class PSO:
    def __init__(self, x0s, num_particles, max_iter, target_error):
        self.particles = [Particle(x0) for x0 in x0s]
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.target_error = target_error
        self.best_global_value = float('inf')
        self.best_global_position = np.array([0, 0 ,0])

    def optimize(self):
        for iteration in range(self.max_iter):
            for particle in self.particles:
                particle.update(self.best_global_position)
                if particle.value < self.best_global_value:
                    self.best_global_value = particle.value
                    self.best_global_position = particle.position
            if abs(target_func(self.best_global_position,y)) < self.target_error:
                break
            iter_fitness = fitness(self.best_global_position,y) 
            print(f'{iteration}: {iter_fitness}')
        return self.best_global_position


    

if __name__=="__main__":
    pred_line_width = input("請輸入想預測的線寬長度，將得到最佳的三個加工參數解：")
    pred_line_width  = int(pred_line_width)
    y = pred_line_width**2

    x0s = [[random.uniform(0, 13), random.uniform(149, 300),random.uniform(50, 70)] for _ in range(10)]
    pso = PSO(x0s, num_particles=1000, max_iter=100, target_error=0.01)
    
    best_position = pso.optimize()
    print("Best solution: ", best_position)
    print("Best fitness value: ", fitness(best_position,y))

    # fitness越小，代表越接近目標函數的最佳解