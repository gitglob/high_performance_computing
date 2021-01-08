import matplotlib.pyplot as plt

time = []
l1_miss = []
li_hit = []
l2_miss = []
l2_hit = []

with open("studio_analysis/all.txt") as f:
    lines = f.readlines()
    time = [float(line.split()[0]) for line in lines]
    l1_hit = [float(line.split()[3]) for line in lines]
    l1_miss = [float(line.split()[4]) for line in lines]
    l2_hit = [float(line.split()[5]) for line in lines]
    l2_miss = [float(line.split()[6]) for line in lines]
    
    
mkn_time = []
mnk_time = []
kmn_time = []
knm_time = []
nmk_time = []
nkm_time = []
mkn_l1_miss = []
mnk_l1_miss = []
kmn_l1_miss = []
knm_l1_miss = []
nmk_l1_miss = []
nkm_l1_miss = []
mkn_l2_miss = []
mnk_l2_miss = []
kmn_l2_miss = []
knm_l2_miss = []
nmk_l2_miss = []
nkm_l2_miss = []
mkn_l1_hit = []
mnk_l1_hit = []
kmn_l1_hit = []
knm_l1_hit = []
nmk_l1_hit = []
nkm_l1_hit = []
mkn_l2_hit = []
mnk_l2_hit = []
kmn_l2_hit = []
knm_l2_hit = []
nmk_l2_hit = []
nkm_l2_hit = []
for i in range(0,36):
    if i%6==0:
        mkn_time.append(time[i])
        mkn_l1_miss.append(l1_miss[i])
        mkn_l1_hit.append(l1_hit[i])
        mkn_l2_miss.append(l2_miss[i])
        mkn_l2_hit.append(l2_hit[i])
    if i%6==1:
        mnk_time.append(time[i])
        mnk_l1_miss.append(l1_miss[i])
        mnk_l1_hit.append(l1_hit[i])
        mnk_l2_miss.append(l2_miss[i])
        mnk_l2_hit.append(l2_hit[i])
    if i%6==2:
        nkm_time.append(time[i])
        nkm_l1_miss.append(l1_miss[i])
        nkm_l1_hit.append(l1_hit[i])
        nkm_l2_miss.append(l2_miss[i])
        nkm_l2_hit.append(l2_hit[i])
    if i%6==3:
        nmk_time.append(time[i])
        nmk_l1_miss.append(l1_miss[i])
        nmk_l1_hit.append(l1_hit[i])
        nmk_l2_miss.append(l2_miss[i])
        nmk_l2_hit.append(l2_hit[i])
    if i%6==4:
        knm_time.append(time[i])
        knm_l1_miss.append(l1_miss[i])
        knm_l1_hit.append(l1_hit[i])
        knm_l2_miss.append(l2_miss[i])
        knm_l2_hit.append(l2_hit[i])
    if i%6==5:
        kmn_time.append(time[i])
        kmn_l1_miss.append(l1_miss[i])
        kmn_l1_hit.append(l1_hit[i])
        kmn_l2_miss.append(l2_miss[i])
        kmn_l2_hit.append(l2_hit[i])

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("CPU time")    
ax1.set_xlabel('m/k/n value (int)')
ax1.set_ylabel('Time(sec)')
x = [10, 50, 100, 500, 1000, 2000]
ax1.plot(x,mkn_time, '--x', c='b', label='mkn')
ax1.plot(x,mnk_time, '-x', c='r', label='mnk')
ax1.plot(x,kmn_time, '-x', c='g', label='kmn')
ax1.plot(x,knm_time, '-x', c='y', label='knm')
ax1.plot(x,nkm_time, '-x', c='c', label='nkm')
ax1.plot(x,nmk_time, '-x', c='k', label='nmk')
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("L1 cache hit")    
ax1.set_xlabel('m/k/n value (int)')
ax1.set_ylabel('L1 hits')
x = [10, 50, 100, 500, 1000, 2000]
ax1.plot(x,mkn_l1_hit, '--x', c='b', label='mkn')
ax1.plot(x,mnk_l1_hit, '-x', c='r', label='mnk')
ax1.plot(x,kmn_l1_hit, '-x', c='g', label='kmn')
ax1.plot(x,knm_l1_hit, '-x', c='y', label='knm')
ax1.plot(x,nkm_l1_hit, '-x', c='c', label='nkm')
ax1.plot(x,nmk_l1_hit, '-x', c='k', label='nmk')
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("L1 cache miss")    
ax1.set_xlabel('m/k/n value (int)')
ax1.set_ylabel('L1 misses')
x = [10, 50, 100, 500, 1000, 2000]
ax1.plot(x,mkn_l1_miss, '--x', c='b', label='mkn')
ax1.plot(x,mnk_l1_miss, '-x', c='r', label='mnk')
ax1.plot(x,kmn_l1_miss, '-x', c='g', label='kmn')
ax1.plot(x,knm_l1_miss, '-x', c='y', label='knm')
ax1.plot(x,nkm_l1_miss, '-x', c='c', label='nkm')
ax1.plot(x,nmk_l1_miss, '-x', c='k', label='nmk')
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("L2 cache hit")    
ax1.set_xlabel('m/k/n value (int)')
ax1.set_ylabel('L2 hits')
x = [10, 50, 100, 500, 1000, 2000]
ax1.plot(x,mkn_l2_hit, '--x', c='b', label='mkn')
ax1.plot(x,mnk_l2_hit, '-x', c='r', label='mnk')
ax1.plot(x,kmn_l2_hit, '-x', c='g', label='kmn')
ax1.plot(x,knm_l2_hit, '-x', c='y', label='knm')
ax1.plot(x,nkm_l2_hit, '-x', c='c', label='nkm')
ax1.plot(x,nmk_l2_hit, '-x', c='k', label='nmk')
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("L2 cache miss")    
ax1.set_xlabel('m/k/n value (int)')
ax1.set_ylabel('L2 misses')
x = [10, 50, 100, 500, 1000, 2000]
ax1.plot(x,mkn_l2_miss, '--x', c='b', label='mkn')
ax1.plot(x,mnk_l2_miss, '-x', c='r', label='mnk')
ax1.plot(x,kmn_l2_miss, '-x', c='g', label='kmn')
ax1.plot(x,knm_l2_miss, '-x', c='y', label='knm')
ax1.plot(x,nkm_l2_miss, '-x', c='c', label='nkm')
ax1.plot(x,nmk_l2_miss, '-x', c='k', label='nmk')
leg = ax1.legend()
plt.show()