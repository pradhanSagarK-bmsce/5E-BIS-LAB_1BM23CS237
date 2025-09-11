import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import sympy as sp
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

POPULATION_SIZE = 150
GENE_HEAD_LENGTH = 5
MAX_ARITY = 2
GENE_TAIL_LENGTH = GENE_HEAD_LENGTH * (MAX_ARITY - 1) + 1
GENE_TOTAL_LENGTH = GENE_HEAD_LENGTH + GENE_TAIL_LENGTH

NUMBER_OF_GENERATIONS = 200
INPUT_VARIABLES = ['x0','x1']
FUNCTION_SET = ['*','/','+','**']
TERMINAL_SET = INPUT_VARIABLES

MUTATION_PROB_INITIAL = 0.3
MUTATION_PROB_FINAL = 0.15
CONST_PERTURB_PROB = 0.6
CONST_PERTURB_STD = 0.05
TOURNAMENT_SIZE_INITIAL = 1
TOURNAMENT_SIZE_FINAL = 5
ELITISM = 2
PARSIMONY_COEFF = 1e-6

def safe_div(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.true_divide(x, y)
        out[~np.isfinite(out)] = 1e6
    return out

def random_constant():
    r = random.random()
    if r < 0.5: return 0.5
    elif r < 0.8: return 1.0
    else: return round(random.uniform(0.01, 5.0),4)

def is_constant_symbol(sym):
    try: float(sym); return True
    except: return False

def generate_random_gene():
    head = [random.choice(FUNCTION_SET + TERMINAL_SET) for _ in range(GENE_HEAD_LENGTH)]
    tail = [random.choice(TERMINAL_SET) if random.random()<0.6 else random_constant() for _ in range(GENE_TAIL_LENGTH)]
    return head + tail

def seed_ke_gene():
    tokens = ['*','*',0.5,'x0','**','x1',2.0]
    while len(tokens) < GENE_TOTAL_LENGTH:
        tokens.append(random.choice(TERMINAL_SET) if random.random()<0.6 else random_constant())
    return tokens

def decode_gene_to_expression(gene, readable=False):
    n = len(gene)
    def build(i):
        if i>=n: return "1e6", n
        token = gene[i]
        if token in FUNCTION_SET:
            left, ni = build(i+1)
            right, ni = build(ni)
            if readable:
                token_map = {'*':'×','/':'÷','+':'+','**':'^'}
                op = token_map[token]
                return f"({left} {op} {right})", ni
            if token=='/': return f"safe_div({left},{right})", ni
            else: return f"({left} {token} {right})", ni
        else:
            return str(float(token)) if is_constant_symbol(token) else str(token), i+1
    try: expr,_ = build(0); return expr
    except: return None

def evaluate_expression_on_data(expr_str, X_data):
    if expr_str is None: return np.full(X_data.shape[0],1e6)
    try:
        env = {f'x{i}': X_data[:,i] for i in range(X_data.shape[1])}
        env['safe_div'] = safe_div
        env['np'] = np
        y_pred = eval(expr_str, {}, env)
        if not np.all(np.isfinite(y_pred)): return np.full(X_data.shape[0],1e6)
        return y_pred
    except:
        return np.full(X_data.shape[0],1e6)

def complexity_of_gene(gene):
    return sum(1 for t in gene if t in FUNCTION_SET) + 0.01*len(gene)

def compute_fitness(gene,X,y):
    expr = decode_gene_to_expression(gene)
    if expr is None: return 1e9
    y_pred = evaluate_expression_on_data(expr,X)
    mse = np.mean((y - y_pred)**2)
    return mse + PARSIMONY_COEFF*complexity_of_gene(gene)

def tournament_select(pop, fitnesses, tour_size):
    best_idx = None
    for _ in range(tour_size):
        i = random.randrange(len(pop))
        if best_idx is None or fitnesses[i]<fitnesses[best_idx]: best_idx=i
    return deepcopy(pop[best_idx])

def mutate_gene(gene, mutation_prob):
    new = deepcopy(gene)
    for i in range(len(new)):
        if random.random()<mutation_prob:
            if i<GENE_HEAD_LENGTH: new[i] = random.choice(FUNCTION_SET+TERMINAL_SET)
            else:
                if is_constant_symbol(new[i]):
                    if random.random()<CONST_PERTURB_PROB:
                        val = max(1e-6,float(new[i]) + random.gauss(0,CONST_PERTURB_STD))
                        new[i] = round(val,6)
                    else: new[i] = random_constant()
                else: new[i] = random.choice(TERMINAL_SET)
    return new

def crossover(g1,g2):
    pt = random.randint(1,GENE_TOTAL_LENGTH-1)
    return g1[:pt]+g2[pt:], g2[:pt]+g1[pt:]

def generate_dataset(n=400,seed=42):
    np.random.seed(seed)
    x0 = np.random.uniform(1,100,n)
    x1 = np.random.uniform(0.1,50,n)
    X = np.vstack((x0,x1)).T
    y = 0.5 * x0 * x1**2
    return X,y

def run_gea():
    X,y = generate_dataset()
    pop = [generate_random_gene() for _ in range(POPULATION_SIZE-3)]
    pop.append(seed_ke_gene())
    pop += [generate_random_gene(), generate_random_gene()]
    fitness_history=[]

    for gen in range(NUMBER_OF_GENERATIONS):
        if gen < 50:
            mutation_prob = 0.3
            tour_size = 1
            elite_count = 0
            inject_random = int(0.05*POPULATION_SIZE)
        else:
            mutation_prob = MUTATION_PROB_FINAL
            tour_size = TOURNAMENT_SIZE_FINAL
            elite_count = ELITISM
            inject_random = 0

        fitnesses = [compute_fitness(g,X,y) for g in pop]
        best_idx = int(np.argmin(fitnesses))
        best_fit = fitnesses[best_idx]
        fitness_history.append(best_fit)
        best_expr_readable = decode_gene_to_expression(pop[best_idx], readable=True)

        if (gen+1)<100 and (gen+1)%10==0:
            print(f"Gen {gen+1:03d} | Best Fitness = {best_fit:.8f} | Expr ≈ {best_expr_readable}")
        elif (gen+1)>=100 and (gen+1)%50==0:
            print(f"Gen {gen+1:03d} | Best Fitness = {best_fit:.8f} | Expr ≈ {best_expr_readable}")

        new_pop = [deepcopy(pop[i]) for i in np.argsort(fitnesses)[:elite_count]]
        while len(new_pop) < POPULATION_SIZE:
            if inject_random > 0:
                new_pop.append(generate_random_gene())
                inject_random -= 1
                continue
            p1 = tournament_select(pop, fitnesses, tour_size)
            p2 = tournament_select(pop, fitnesses, tour_size)
            c1, c2 = crossover(p1,p2)
            new_pop.append(mutate_gene(c1, mutation_prob))
            if len(new_pop)<POPULATION_SIZE: new_pop.append(mutate_gene(c2, mutation_prob))
        pop = new_pop

        if best_fit < 1e-6:
            print(f"Early stop at generation {gen+1}, fitness {best_fit:.8e}")
            break

    best_final_idx = int(np.argmin([compute_fitness(g,X,y) for g in pop]))
    best_gene = pop[best_final_idx]
    best_expr = decode_gene_to_expression(best_gene)
    print("\n=== FINAL RESULT ===")
    print(f"Final best Fitness = {compute_fitness(best_gene,X,y):.12f}")
    print("Discovered expression:", best_expr)

    m, v = sp.symbols('m v')
    sympy_expr = sp.sympify(best_expr.replace('x0','m').replace('x1','v'))
    print("SymPy LaTeX:", sp.latex(sympy_expr))

    y_pred = evaluate_expression_on_data(best_expr, X)
    plt.figure(figsize=(18,5))

    plt.subplot(1,3,1)
    errors = np.abs(y - y_pred)
    scatter = plt.scatter(y, y_pred, s=18, c=errors, cmap='plasma', alpha=0.8, edgecolors='none')
    mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    plt.xlabel("True KE")
    plt.ylabel("Predicted KE")
    plt.title("True vs Predicted KE")
    plt.colorbar(scatter, label='Absolute Error')

    plt.subplot(1,3,2)
    generations = np.arange(1, len(fitness_history)+1)
    plt.plot(generations, fitness_history, marker='o', markersize=5, linewidth=2, color='teal', label='Best Fitness')
    plt.fill_between(generations, fitness_history, color='teal', alpha=0.2)
    plt.yscale('log')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (log scale)")
    plt.title("Fitness Progression")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.subplot(1,3,3)
    plt.hist(errors, bins=30, color='coral', alpha=0.7, edgecolor='k')
    plt.xlabel("Absolute Error")
    plt.ylabel("Frequency")
    plt.title("Error Distribution")

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    run_gea()
