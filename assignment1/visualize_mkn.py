import matplotlib.pyplot as plt

y_mkn = []
x_mkn = []

# no optimization
with open("fastest/data_adj.txt") as f:
    lines = f.readlines()
    for line in lines:
        if line == '\n':
            continue
        if line.split()[4] == "matmult_mkn":
            y_mkn.append(float(line.split()[1]))
            x_mkn.append(float(line.split()[0]))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc -g -O3 -ffast-math -funroll-loops")     
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mkn[0:16],y_mkn[0:16], '-x', c='b', label='mkn data')
ax1.axvline(x=32)
ax1.axvline(x=256)
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc -g -O3 -ffast-math -funroll-loops")   
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mkn[20:27],y_mkn[20:27], '-x', c='b', label='mkn data')
ax1.axvline(x=25600)
leg = ax1.legend()
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc -g -O3 -ffast-math -funroll-loops")  
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mkn,y_mkn, '-x', c='b', label='mkn data')
ax1.axvline(x=32)
ax1.axvline(x=256)
ax1.axvline(x=25600)
leg = ax1.legend()
plt.show()

x = [5, 10, 20, 25, 30,
     35, 50, 60, 70, 80,
     90, 100, 110, 120, 150,
     200, 300, 400, 600, 800,
     850, 900, 1000, 1200, 1500,
     1800, 2000, 2400, 2800]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("mkn value - Performance : gcc -g -O3 -ffast-math -funroll-loops")     
ax1.set_ylabel('Performance (Mflops/s)')
ax1.set_xlabel('m/n/k value (int)')
ax1.plot(x,y_mkn, '-x', c='b', label='mkn data')
leg = ax1.legend()
plt.show()