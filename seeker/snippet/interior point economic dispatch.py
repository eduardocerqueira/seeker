#date: 2025-09-26T17:09:29Z
#url: https://api.github.com/gists/cbf427ea89bd1fa4ad1806436f3c6cdd
#owner: https://api.github.com/users/59c1ear73-wq

import numpy as np
from scipy.sparse import lil_matrix, vstack, diags,eye,bmat

#参数设置
a = np.array([0.12, 0.17, 0.15, 0.19])
b = np.array([14.8, 16.57, 15.55, 16.21])
c = np.array([89, 83, 100, 70])
max_iter = 50  #迭代次数

eta_c = 0.95
eta_d = 0.9
SOC_ini = 100
SOC_max = 300
SOC_min = 60
Pc_max= Pd_max = 0.2 * SOC_max
Pc_min = Pd_min = 0.2 * SOC_min
G = 4
T = 24
VWC = 50
Pg_min = np.array([28, 20, 30, 20])
Pg_max = np.array([200, 290, 190, 260])
RU = np.array([40, 30, 30, 50])
RD = np.array([40, 30, 30, 50])

Load = np.array([510, 530, 516, 510, 515, 544, 646, 686, 741, 734, 748, 760, 754, 700, 686, 720, 714, 761, 727, 714, 618, 584, 578, 544])
W = np.array([44.1, 48.5, 65.7, 144.9, 202.3, 317.3, 364.4, 317.3, 271, 306.9, 424.1, 398, 487.6, 521.9, 541.3, 560, 486.8, 372.6, 367.4, 314.3, 316.6, 311.4, 405.4, 470.4])

#变量索引
nPg = G * T
nSOC = T
nPd = T
nPc = T
nPw = T
nPwc = T
n_var = G * T + 5 * T
#利用词典分配索引
var_index = {}
i = 0;
var_index['Pg'] = i#先T循环后G循环
i+= nPg

var_index['SOC'] = i
i+= nSOC

var_index['Pd'] = i
i+= nPd

var_index['Pc'] = i
i+= nPc

var_index['Pw'] = i
i+= nPw

var_index['Pwc'] = i

#建立变量
var = np.ones(n_var)
start = var_index['Pg']
for g in range(G):
    Pg = start + g * T
    var[Pg:Pg+T] = 0.5 * (Pg_max[g] + Pg_min[g])
start = var_index['SOC']
var[start:start+T] = 0.5 * (SOC_max + SOC_min)
start = var_index['Pd']
var[start:start+T] = 0.5 * (Pd_max + Pd_min)
start = var_index['Pc']
var[start:start+T] = 0.5 * (Pc_max + Pc_min)
start = var_index['Pw']
var[start:start+T] = 0.5 * W
start = var_index['Pwc']
var[start:start+T] = 0.5 * W

#统计约束条件数量
n_ub = G * T * 3  + T * 3 - 2 * G
n_lb = G * T + T * 3
n_ineq = n_ub + n_lb
n_eq = T * 3 + 1

#松弛变量与拉格朗日乘子
l = np.ones(n_lb) #下限松弛变量
L = diags(l)
z = np.ones(n_lb) #下限拉格朗日乘子
Z = diags(z)
u = np.ones(n_ub) #上限松弛变量
U = diags(u)
w = np.ones(n_ub) #上限拉格朗日乘子
W = diags(w)
y = np.zeros(n_eq) #等式约束拉格朗日乘子

#上限约束索引
ub_index = {}
i = 0
ub_index['Pgmax'] = i
i+= nPg
ub_index['SOCmax'] = i
i+= nSOC
ub_index['Pdmax'] = i
i+= nPd
ub_index['Pcmax'] = i
i+= nPc
ub_index['Rampup'] = i
i+= G * (T-1)
ub_index['Rampdown'] = i
i+= G * (T-1)
#下限约束索引
lb_index = {}
i = 0
lb_index['Pgmin'] = i
i+= nPg
lb_index['SOCmin'] = i
i+= nSOC
lb_index['Pdmin'] = i
i+= nPd
lb_index['Pcmin'] = i
i+= nPc
#等式约束索引
eq_index = {}
i = 0
eq_index['SOC0'] = i
i+= 1
eq_index['SOCt'] = i
i+= T - 1
eq_index['SOCini'] = i
i+= 1
eq_index['Load_balance'] = i
i+= T
eq_index['Wind_balance'] = i
i+= T
#求7-72矩阵中各个元素
#先求H
H = lil_matrix((n_var, n_var))
start_row = var_index['Pg']
start_col = var_index['Pg']
for g in range (G):
    for t in range (T):
        row = start_row + t
        col = start_col + t
        H[row, col] = -2 * a[g]
    start_row += T
    start_col += T
##然后求等式约束的梯度g（x），这里书本中默认转置了一下
nabla_eq =  lil_matrix((n_eq, n_var))
#等式约束SOC0
nabla_eq[eq_index['SOC0'], var_index['SOC']] = 1.0
nabla_eq[eq_index['SOC0'], var_index['Pc']] = -eta_c
nabla_eq[eq_index['SOC0'], var_index['Pd']] = 1/eta_d
#等式约束SOCt
start_row = eq_index['SOCt']
start_col = var_index['SOC']
for t in range (1,T-1):
    row = start_row + t - 1
    col = start_col + t - 1
    nabla_eq[row, col] = -1
    nabla_eq[row, col+1] = 1

start_row = eq_index['SOCt']
start_col = var_index['Pc']
for t in range (1,T-1):
    row = start_row + t - 1
    col = start_col + t
    nabla_eq[row, col] = -eta_c

start_row = eq_index['SOCt']
start_col = var_index['Pd']
for t in range (1,T-1):
    row = start_row + t - 1
    col = start_col + t
    nabla_eq[row, col] = 1/eta_d

#等式约束SOC24
start_row = eq_index['SOCini']
start_col = var_index['SOC']
row = start_row
col = start_col + T - 1
nabla_eq[row, col] = 1

#等式约束Load balance
start_row = eq_index['Load_balance']
start_col = var_index['Pw']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_eq[row, col] = 1

start_row = eq_index['Load_balance']
start_col = var_index['Pg']
for t in range (T):
    for g in range (G):
        row = start_row + t
        col = start_col + t + g * T
        nabla_eq[row, col] = 1

start_row = eq_index['Load_balance']
start_col = var_index['Pd']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_eq[row, col] = 1

start_row = eq_index['Load_balance']
start_col = var_index['Pc']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_eq[row, col] = -1

#等式约束Wind_balance
start_row = eq_index['Wind_balance']
start_col = var_index['Pw']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_eq[row, col] = 1

start_row = eq_index['Wind_balance']
start_col = var_index['Pwc']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_eq[row, col] = 1


##不等式约束
nabla_ub =  lil_matrix((n_ub, n_var))
nabla_lb =  lil_matrix((n_lb, n_var))
#上限约束Pgmax
start_row = ub_index['Pgmax']
start_col =var_index['Pg']
for g in range (G):
    for t in range (T):
        row = start_row + t
        col = start_col + t
        nabla_ub[row, col] = 1
    start_row += T
    start_col += T


#上限约束SOCmax
start_row = ub_index['SOCmax']
start_col = var_index['SOC']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_ub[row, col] = 1

#上限约束Pdmax
start_row = ub_index['Pdmax']
start_col = var_index['Pd']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_ub[row, col] = 1

#上限约束Pcmax
start_row = ub_index['Pcmax']
start_col = var_index['Pc']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_ub[row, col] = 1

#上限约束Rampup
start_row = ub_index['Rampup']
start_col = var_index['Pg']
for g in range (G):
    for t in range (1,T):
        row = start_row + t - 1
        col = start_col + t
        nabla_ub[row, col] = 1
        nabla_ub[row, col-1] = -1
    start_row += T - 1
    start_col += T

#上线约束Rampdown
start_row = ub_index['Rampdown']
start_col = var_index['Pg']
for g in range (G):
    for t in range (1,T):
        row = start_row + t - 1
        col = start_col + t
        nabla_ub[row, col] = -1
        nabla_ub[row, col - 1] = 1
    start_row += T - 1
    start_col += T

#下限约束Pgmin
start_row = lb_index['Pgmin']
start_col = var_index['Pg']
for g in range (G):
    for t in range (T):
        row = start_row + t
        col = start_col + t
        nabla_lb[row, col] = 1
    start_row += T
    start_col += T

#下限约束SOCmin
start_row = lb_index['SOCmin']
start_col = var_index['SOC']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_lb[row, col] = 1

#下限约束Pdmin
start_row = lb_index['Pdmin']
start_col = var_index['Pd']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_lb[row, col] = 1

#下限约束Pcmin
start_row = lb_index[('Pcmin')]
start_col = var_index['Pc']
for t in range (T):
    row = start_row + t
    col = start_col + t
    nabla_lb[row, col] = 1

#将ub、lb合成 不等式约束nabla_ineq
nabla_ineq = lil_matrix((n_ineq, n_var))
nabla_ineq = vstack([nabla_ub, nabla_lb])

#将分块矩阵按照7-72拼接为一个大矩阵LHS
I_lb = eye(n_lb)
I_ub = eye(n_ub)
LHS = lil_matrix((n_var+n_eq+2*(n_ub+n_lb),n_var+n_eq+2*(n_ub+n_lb)))
O = None
LHS = bmat([[H,nabla_eq.T,nabla_lb.T,nabla_ub.T,O,O],
           [nabla_eq,O,O,O,O,O],
           [nabla_lb,O,O,O,-I_lb,O],
           [nabla_ub,O,O,O,O,I_ub],
           [O,O,L,O,Z,O],
           [O,O,O,U,O,W]]
           )

#下面根据7-56~61定义7-72中的右侧矩阵RHS
#f(x)梯度nabla_obj
nabla_obj = np.ones(n_var)
start = var_index['Pg']
for g in range (G):
    for t in range (T):
        nabla_obj[start+t] = 2 * a[g] * var[var_index['Pg']] + b[g]
    start += T

start = var_index[('Pwc')]
for t in range (T):
    nabla_obj[start+t] = VWC

##  定义RHS
#定义Lx 7-56
Lx = nabla_obj - nabla_eq @ y - nabla_lb @ z - nabla_ub @ w

#定义Ly 7-57
Ly = np.ones(n_eq)
Ly_SOC0 = var[var_index['SOC']] - SOC_ini - var[var_index['Pc']] * eta_c +var[var_index['Pd']] * 1/eta_d
Ly_SOCt = (var[var_index['SOC']+1:var_index['SOC']+T] - var[var_index['SOC']:var_index['SOC']+T-1]
           - var[var_index['Pc']+1:var_index['Pc']+T] * eta_c +var[var_index['Pd']+1:var_index['Pd']+T] * 1/eta_d)
Ly_SOC24 = var[var_index['SOC']+T-1] - SOC_ini
Pg_t = var[var_index['Pg']:var_index['SOC']].reshape(G, T)
sum_pg = np.ones(T)
for t in range (T):
    sum_pg_t = np.sum(Pg_t[:, t])
    sum_pg[t] = sum_pg_t
Ly_demand_balance = (var[var_index['Pw']:var_index['Pw']+T] + sum_pg + var[var_index['Pd']:var_index['Pd']+T]
                     - var[var_index['Pc']:var_index['Pc']+T] - Load)
Ly_wind_balance = var[var_index['Pw']:var_index['Pw']+T] + var[var_index['Pwc']:var_index['Pwc']+T] - W
Ly = vstack([Ly_SOC0, Ly_SOCt, Ly_SOC24,Ly_demand_balance,Ly_wind_balance])#垂直拼接

#定义Lz 7-58
for g in range (G):
    Lz_pgmin = var[var_index['Pg'] + T * g:var_index['Pg'] + T * (g + 1)] - l[lb_index['Pgmin']:lb_index['SOCmin']] - Pg_min[g]
Lz_SOCmin = var[var_index['SOC']:var_index['SOC']+T] - SOC_min - l[lb_index['SOCmin']:lb_index['Pdmin']]
Lz_Pdmin = var[var_index['Pd']:var_index['Pd']+T] - Pd_min - l[lb_index['Pdmin']:lb_index['Pcmin']]
Lz_Pcmin = var[var_index['Pc']:var_index['Pc']+T] - Pc_min - l[lb_index['Pcmin']:lb_index['Pcmin']+T]
Lz = vstack(Lz_pgmin, Lz_SOCmin, Lz_Pdmin, Lz_Pcmin)

#定义Lw 7-59
for g in range (G):
    Lw_pgmax = var[var_index['Pg'] + T * g:var_index['Pg'] + T * (g + 1)] + u[ub_index['Pgmax']:ub_index['SOCmax']] - Pg_max[g]
Lw_SOCmax = var[var_index['SOC']:var_index['SOC']+T] - SOC_max + u[ub_index['SOCmax']:ub_index['Pdmax']]
Lw_Pdmax = var[var_index['Pd']:var_index['Pd']+T] - Pd_max + u[ub_index['Pdmin']:ub_index['Pcmax']]
Lw_Pcmax = var[var_index['Pc']:var_index['Pc']+T] - Pc_min + u[ub_index['Pcmax']:ub_index['Rampup']]
for g in range (G):
    Lw_Rampup = (var[var_index['Pg']+1+(T-1)*g:var_index['Pg']+1+(T-1)*(g+1)]
                 - var[var_index['Pg']+(T-1)*g:var_index['Pg']+(T-1)*(g+1)] - RU[g] + u[ub_index['Rampup']:ub_index['Rampdown']])
    Lw_Rampdown =  (-var[var_index['Pg']+1+(T-1)*g:var_index['Pg']+1+(T-1)*(g+1)]
                 + var[var_index['Pg']+(T-1)*g:var_index['Pg']+(T-1)*(g+1)] - RD[g] + u[ub_index['Rampdown']:ub_index['Rampdown']+G*(T-1)])
Lw = vstack(Lw_pgmax, Lw_SOCmax, Lw_Pdmax, Lw_Pcmax,Lw_Rampup, Lw_Rampdown)
#定义对偶间隙Gap、扰动因子mu
Gap = l@z-u@w
mu = 0.1 * Gap / (2*n_ineq)
#定义Lmu_l 7-60
Lmu_l = l@z - mu
#定义Lmu_u 7-61
Lmu_u = u@w + mu







