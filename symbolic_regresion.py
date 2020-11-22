import numpy as np
import matplotlib.pyplot as plt
from itertools import zip_longest
import copy
import json


class Individual:
    def __init__(self, sym_reg_config=None, recursion=10):
        self.nodes = 0
        
        self.operation = "+"
    
        if sym_reg_config is not None:
            operation_index = np.random.randint(len(sym_reg_config["operations"]))
            self.operation = sym_reg_config["operations"][operation_index]
        self.nodes += 1

        self.operands = []
        if sym_reg_config is not None:
            self.random_operands(sym_reg_config, operation_index, recursion)

        self.evaluation_string = ""
        if sym_reg_config is not None:
            self.evaluation_string_update()

        self.sse = 0

        if sym_reg_config is not None:
            self.points_x = sym_reg_config["points_x"]
            self.points_y = sym_reg_config["points_y"]
            self.constants_values = sym_reg_config["constants_values"]


    def random_operands(self, sym_reg_config, operation_index, recursion):
        operand_max_amount = sym_reg_config["operands_max_amount"][operation_index]
        if operand_max_amount > 1:
            operand_amount = np.random.randint(2, operand_max_amount)
        elif operand_max_amount == 1:
            operand_amount = 1

        if operand_amount == 1:                         # if operand is function that take only one parameter
            operand_type = np.random.randint(recursion)
            if operand_type <= recursion*0.2:
                operation_index = np.random.randint(len(sym_reg_config["variables"]))
                operand = sym_reg_config["variables"][operation_index]
                self.nodes += 1
            elif operand_type <= recursion:
                operand = Individual(sym_reg_config, recursion=max([1, recursion-1]))
                self.nodes += operand.nodes
            self.operands.append(operand)

        else:
            for i in range(operand_amount):
                if i == 0:
                    operand_type = recursion
                else:
                    operand_type = np.random.randint(recursion)

                if operand_type <=recursion*0.1:
                    operation_index = np.random.randint(len(sym_reg_config["constants_values"]))
                    operand = sym_reg_config["constants_values"][operation_index]
                    self.nodes+=1
                elif operand_type <= recursion*0.4:
                    operation_index = np.random.randint(len(sym_reg_config["variables"]))
                    operand = sym_reg_config["variables"][operation_index]
                    self.nodes+=1
                elif operand_type <= recursion:
                    operand = Individual(sym_reg_config, recursion=max([1, recursion-1]))
                    self.nodes+=operand.nodes
                self.operands.append(operand)
            

    def nodes_number_update(self, acc=0):
        self.nodes = 1   # for operation
        #print(acc, "xD", self.evaluation_string)
        for i in range(len(self.operands)):
            #print(len(self.operands))
            if type(self.operands[i]) == type(self):
                #print(self.operands[i], self)
                # self.operands[i].evaluation_string_update()
                #print(acc, self.operands[i].evaluation_string)
                self.operands[i].nodes_number_update(acc+1)
                self.nodes+=self.operands[i].nodes
            else:
                self.nodes+=1   # for operation
        #print(acc, "xDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")


    def evaluation_string_update(self):
        self.evaluation_string = "("
        if len(self.operands) == 1:
            arg_id = self.operation.find(')')
            if type(self.operands[0]) == type(self):
                    self.operands[0].evaluation_string_update()
                    self.evaluation_string += self.operation[:arg_id] + self.operands[0].evaluation_string + self.operation[arg_id:]
            else:
                self.evaluation_string += self.operation[:arg_id] + self.operands[0] + self.operation[arg_id:]
        else:
            for i in range(len(self.operands)):
                if type(self.operands[i]) == type(self):
                    self.operands[i].evaluation_string_update()
                    self.evaluation_string += self.operands[i].evaluation_string
                else:
                    self.evaluation_string += self.operands[i]
                    
                if i != len(self.operands)-1:
                    self.evaluation_string += self.operation

        self.evaluation_string += ")"


    def mutation(self):
        sub_substitute = Individual(sym_reg_config, 3)
        n = np.random.randint(self.nodes)
        self.get_and_change_sub_individual(n, nodes=self.nodes, sub_substitute=sub_substitute)


    def get_and_change_sub_individual(self, n, nodes, level=1, sub_substitute=None):
        if n > nodes:
            #print("XSXSXSXSXSXSXSXS")
            n = nodes

        if level == n:
            sub = copy.copy(self)
            #print(level, sub.evaluation_string)
            if sub_substitute is not None:
                self = sub_substitute
            return sub, level
            
        for i in range(len(self.operands)):
            level+=1
            #print(level)

            if type(self.operands[i]) == type(self):        # if operand is object of Individual class
                
                if level == n:
                    sub = copy.copy(self.operands[i])
                    #print(level, sub)
                    if sub_substitute is not None:
                        self.operands[i] = sub_substitute
                    return sub, level
                else:
                    sub, sublevel = self.operands[i].get_and_change_sub_individual(n, nodes, level, sub_substitute)

                    level+=(sublevel-level)
                    #print(level, sub)
                    if sub != None:
                        if sub_substitute is not None:
                            self.operands[i] = sub_substitute
                        return sub, level
            
            else:                                           # if operand isn't object of Individual class
                if level == n:
                    sub = copy.copy(self.operands[i])
                    #print(level, sub)
                    if sub_substitute is not None:
                        self.operands[i] = sub_substitute
                    return sub, level

        return None, level


    def evaluate(self, x):
        a = self.constants_values[0]
        b = self.constants_values[1]
        c = self.constants_values[2]
        d = self.constants_values[3]
        result = eval(self.evaluation_string)
        return result


    def sum_squared_error(self):
        self.sse = 0
        
        for i in range(len(self.points_x)):
            self.sse += (self.points_y[i] - self.evaluate(self.points_x[i]))**2

        return self.sse


    def copypaste(self):
        individual = Individual(sym_reg_config, 1)
        operation = copy.copy(self.operation)
        operands = []

        for i in range(len(self.operands)):
            if type(self.operands[i]) == type(self):
                operand = self.operands[i].copypaste()
            else:
                operand = self.operands[i]
            
            operands.append(operand)

        individual.operation        = operation
        individual.operands         = operands
        individual.points_x         = self.points_x
        individual.points_y         = self.points_y
        individual.constants_values = self.constants_values
        individual.nodes            = self.nodes
        individual.sse              = self.sse
        return individual


def create_initial_population(sym_reg_config):
    population = []
    for i in range(sym_reg_config["N"]):
        population.append(Individual(sym_reg_config))
        print(population[-1].evaluation_string)

    return population
        

def fit_function(population):
    for individual in population:
        individual.sum_squared_error()


def find_the_best_individual(population):
    the_best_individual = population[0]
    for i in range(1, len(population)):
        if the_best_individual.sse > population[i].sse:
            the_best_individual = population[i]

    the_best_individual = the_best_individual.copypaste()
    the_best_individual.evaluation_string_update()
    return the_best_individual


def tournament(population):
    parent_population = []
    population_copy = population.copy()

    #print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    for i in range(len(population_copy)):
        if len(population_copy) > 1:
            #number_of_competitors = np.random.randint(1, len(population_copy))
            number_of_competitors = 4
            #print(number_of_competitors)
            r = np.arange(0, len(population), 1)
            np.random.shuffle(r)
            competitors = []
            for j in range(number_of_competitors):
                competitors.append(population_copy[r[j]])
            winner = find_the_best_individual(competitors)
            winner = winner.copypaste()
            winner.evaluation_string_update()
            parent_population.append(winner)
            #population_copy.remove(parentPopulation[-1])
        else:
            parent_population.append(parent_population[-1].copypaste())
            #population_copy.clear()
        # print(parent_population[-1].evaluation_string)

    return parent_population

def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def crossover(population):
    parent_population = tournament(population)

    np.random.shuffle(parent_population)
    pairs = grouper(2, parent_population)
    offspring_population = []

    copy_or_offspring_probability = 30
    # create two offSprings for each pair
    for pair in pairs:
        copy_or_offspring = np.random.randint(0, 100)
        if copy_or_offspring < copy_or_offspring_probability:
            offspring_population.append(pair[0])
            offspring_population.append(pair[1])
        else:
            node0 = np.random.randint(1, pair[0].nodes)
            node1 = np.random.randint(1, pair[1].nodes)

            sub_individual0, _ = pair[0].get_and_change_sub_individual(node0, pair[0].nodes)
            if type(sub_individual0)==type(pair[0]):
                sub_individual0 = sub_individual0.copypaste()
                sub_individual0.evaluation_string_update()

            sub_individual1, _ = pair[1].get_and_change_sub_individual(node1, pair[1].nodes, sub_substitute=sub_individual0)
            if type(sub_individual1)==type(pair[0]):
                sub_individual1 = sub_individual1.copypaste()
                sub_individual1.evaluation_string_update()

            _              , _ = pair[0].get_and_change_sub_individual(node0, pair[0].nodes, sub_substitute=sub_individual1)

            offspring_population.append(pair[0])
            offspring_population.append(pair[1])

    return offspring_population


def mutation(population):
    for individual in population:
        mutationProbability = 1/individual.nodes
        # print(mutationProbability)
        mutate = np.random.randint(0, 10000) / 10000
        if mutate < mutationProbability:
            individual.mutation()
    return population


def population_substitute(population, offspring_population):
    copy_probability = 30
    new_population = []
    for i in range(len(population)):
        r = np.random.randint(0, 100)
        if r < copy_probability:
            new_population.append(population[i])
        else:
            new_population.append(offspring_population[i])

    return new_population


def evolution_plot(iterations, the_best_individuals, title, ylabel):
    i = np.arange(1, iterations+1)
    sses = [the_best_individual.sse for the_best_individual in the_best_individuals]
    # print(i)
    # print(sses)
    plt.plot(i, sses)
    plt.grid()

    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel(ylabel)
    plt.xlim(i[0], i[-1])

    plt.show()


def symbolic_regression(sym_reg_config):
    population = create_initial_population(sym_reg_config)
    fit_function(population)

    the_best_individuals_local = []
    the_best_individuals_global = []
    iterations = 0
    for i in range(sym_reg_config["iterations"]):
        iterations+=1
        print("iteration: ", i+1)
        #print("crossover")

        parent_population = tournament(population)
        offspring_population = crossover(parent_population)
        population = population_substitute(population, offspring_population)

        for individual in population:
            individual.nodes_number_update()
            individual.evaluation_string_update()

        #print("mutation")
        population = mutation(population)

        for individual in population:
            individual.nodes_number_update()
            individual.evaluation_string_update()
        
        fit_function(population)
        the_best_individual = find_the_best_individual(population)
        # print(the_best_individual)
        # print(the_best_individual.evaluation_string)
        # print(the_best_individual.sse)
        the_best_individuals_local.append(the_best_individual)
        
        print("local sse:", the_best_individuals_local[-1].sse)
        if i == 0:
            the_best_individuals_global.append(the_best_individual)
        else:
            if the_best_individuals_global[-1].sse < the_best_individual.sse:
                the_best_individuals_global.append(the_best_individuals_global[-1])
            else:
                the_best_individuals_global.append(the_best_individual)

        print("global sse:", the_best_individuals_global[-1].sse)
        print("evaluation_string:", the_best_individual.evaluation_string)

        if the_best_individual.sse <= sym_reg_config["error"]:
            break
    
    evolution_plot(iterations, the_best_individuals_local, "sum squared error of the best individual in each epoch individualy","sum squared error")
    evolution_plot(iterations, the_best_individuals_global, "sum squared error of the best individual in all previous epoch","sum squared error")

    return the_best_individuals_global[-1]


def plot_approximation(x, individual):
    y = [individual.evaluate(xi) for xi in x]
    plt.plot(x, y, label="approximation")

    plt.title("Approximated function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(x[0], x[-1])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Editable parameters
    data_path = 'data.json'
    json_file = open(data_path)
    data = json.load(json_file)
    x = data["x"]
    y = data["y"]
    
    sym_reg_config = {
        "operations": ["+", "-", "*", "np.cos()", "np.sin()", "np.sqrt(np.abs())"],
        "operands_max_amount": [4, 4, 4, 1, 1, 1],
        "constants_values": [str(i/10000) for i in np.squeeze(np.random.randint(0, 100000, (1, 4)))],
        "variables": ["x"],
        "N": 50,
        "error": 0.01,
        "iterations": 500,
        "points_x": x,
        "points_y": y,
    }

    # Symbolic regression
    the_best_individual = symbolic_regression(sym_reg_config)
    
    # Show results
    x_lin = np.linspace(min(x)*1.25, max(x)*1.25, 1000)
    plt.plot(x, y, "o", label="points")
    plt.grid()
    plot_approximation(x_lin, the_best_individual)