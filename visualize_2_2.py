import matplotlib.pyplot as plt

y_mnk = []
x_mnk = []
y_mkn = []
x_mkn = []
y_nkm = []
x_nkm = []
y_nmk = []
x_nmk = []
y_kmn = []
x_kmn = []
y_knm = []
x_knm = []

# no optimization
with open("outputs/per_8694617.txt") as f:
    lines = f.readlines()
    for line in lines:
        if line == '\n':
            continue
        if line.split()[4] == "matmult_mnk":
            y_mnk.append(float(line.split()[0]))
            x_mnk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_mkn":
            y_mkn.append(float(line.split()[0]))
            x_mkn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nmk":
            y_nmk.append(float(line.split()[0]))
            x_nmk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nkm":
            y_nkm.append(float(line.split()[0]))
            x_nkm.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_kmn":
            y_kmn.append(float(line.split()[0]))
            x_kmn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_knm":
            y_knm.append(float(line.split()[0]))
            x_knm.append(float(line.split()[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc")    
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mnk,y_mnk, '-x', c='b', label='mnk data')
ax1.plot(x_mkn,y_mkn, '-x', c='r', label='mkn data')
ax1.plot(x_nmk,y_nmk, '-x', c='g', label='nmk data')
ax1.plot(x_nkm,y_nkm, '-x', c='c', label='nkm data')
ax1.plot(x_kmn,y_kmn, '-x', c='k', label='kmn data')
ax1.plot(x_knm,y_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

x = [5,10,20,50,75,100,200,500,1000,2000]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("mkn value - Memory : gcc")    
ax1.set_ylabel('Memory Footprint (kbytes)')
ax1.set_xlabel('m/n/k value (int)')
ax1.plot(x,x_mnk, '-x', c='b', label='mnk data')
ax1.plot(x,x_mkn, '-x', c='r', label='mkn data')
ax1.plot(x,x_nmk, '-x', c='g', label='nmk data')
ax1.plot(x,x_nkm, '-x', c='c', label='nkm data')
ax1.plot(x,x_kmn, '-x', c='k', label='kmn data')
ax1.plot(x,x_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

y_mnk = []
x_mnk = []
y_mkn = []
x_mkn = []
y_nkm = []
x_nkm = []
y_nmk = []
x_nmk = []
y_kmn = []
x_kmn = []
y_knm = []
x_knm = []

# o3 optimization
with open("O3_optimized_outputs/per_8691920.txt") as f:
    lines = f.readlines()
    for line in lines:
        if line == '\n':
            continue
        if line.split()[4] == "matmult_mnk":
            y_mnk.append(float(line.split()[0]))
            x_mnk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_mkn":
            y_mkn.append(float(line.split()[0]))
            x_mkn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nmk":
            y_nmk.append(float(line.split()[0]))
            x_nmk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nkm":
            y_nkm.append(float(line.split()[0]))
            x_nkm.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_kmn":
            y_kmn.append(float(line.split()[0]))
            x_kmn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_knm":
            y_knm.append(float(line.split()[0]))
            x_knm.append(float(line.split()[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc O3")    
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mnk,y_mnk, '-x', c='b', label='mnk data')
ax1.plot(x_mkn,y_mkn, '-x', c='r', label='mkn data')
ax1.plot(x_nmk,y_nmk, '-x', c='g', label='nmk data')
ax1.plot(x_nkm,y_nkm, '-x', c='c', label='nkm data')
ax1.plot(x_kmn,y_kmn, '-x', c='k', label='kmn data')
ax1.plot(x_knm,y_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

x = [5,10,20,50,75,100,200,500,1000,2000]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("mkn value - Memory : gcc O3")    
ax1.set_ylabel('Memory Footprint (kbytes)')
ax1.set_xlabel('m/n/k value (int)')
ax1.plot(x,x_mnk, '-x', c='b', label='mnk data')
ax1.plot(x,x_mkn, '-x', c='r', label='mkn data')
ax1.plot(x,x_nmk, '-x', c='g', label='nmk data')
ax1.plot(x,x_nkm, '-x', c='c', label='nkm data')
ax1.plot(x,x_kmn, '-x', c='k', label='kmn data')
ax1.plot(x,x_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

y_mnk = []
x_mnk = []
y_mkn = []
x_mkn = []
y_nkm = []
x_nkm = []
y_nmk = []
x_nmk = []
y_kmn = []
x_kmn = []
y_knm = []
x_knm = []

# Ofast
with open("Ofast_optimized_outputs/per_8697294.txt") as f:
    lines = f.readlines()
    for line in lines:
        if line == '\n':
            continue
        if line.split()[4] == "matmult_mnk":
            y_mnk.append(float(line.split()[0]))
            x_mnk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_mkn":
            y_mkn.append(float(line.split()[0]))
            x_mkn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nmk":
            y_nmk.append(float(line.split()[0]))
            x_nmk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nkm":
            y_nkm.append(float(line.split()[0]))
            x_nkm.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_kmn":
            y_kmn.append(float(line.split()[0]))
            x_kmn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_knm":
            y_knm.append(float(line.split()[0]))
            x_knm.append(float(line.split()[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc O3")    
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mnk,y_mnk, '-x', c='b', label='mnk data')
ax1.plot(x_mkn,y_mkn, '-x', c='r', label='mkn data')
ax1.plot(x_nmk,y_nmk, '-x', c='g', label='nmk data')
ax1.plot(x_nkm,y_nkm, '-x', c='c', label='nkm data')
ax1.plot(x_kmn,y_kmn, '-x', c='k', label='kmn data')
ax1.plot(x_knm,y_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

x = [5,10,20,50,75,100,200,500,1000,2000]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("mkn value - Memory : gcc O3")    
ax1.set_ylabel('Memory Footprint (kbytes)')
ax1.set_xlabel('m/n/k value (int)')
ax1.plot(x,x_mnk, '-x', c='b', label='mnk data')
ax1.plot(x,x_mkn, '-x', c='r', label='mkn data')
ax1.plot(x,x_nmk, '-x', c='g', label='nmk data')
ax1.plot(x,x_nkm, '-x', c='c', label='nkm data')
ax1.plot(x,x_kmn, '-x', c='k', label='kmn data')
ax1.plot(x,x_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

y_mnk = []
x_mnk = []
y_mkn = []
x_mkn = []
y_nkm = []
x_nkm = []
y_nmk = []
x_nmk = []
y_kmn = []
x_kmn = []
y_knm = []
x_knm = []

# O3 fully optimized
with open("fully_optimized_O3_outputs/per_8698065.txt") as f:
    lines = f.readlines()
    for line in lines:
        if line == '\n':
            continue
        if line.split()[4] == "matmult_mnk":
            y_mnk.append(float(line.split()[0]))
            x_mnk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_mkn":
            y_mkn.append(float(line.split()[0]))
            x_mkn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nmk":
            y_nmk.append(float(line.split()[0]))
            x_nmk.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_nkm":
            y_nkm.append(float(line.split()[0]))
            x_nkm.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_kmn":
            y_kmn.append(float(line.split()[0]))
            x_kmn.append(float(line.split()[1]))
        elif line.split()[4] == "matmult_knm":
            y_knm.append(float(line.split()[0]))
            x_knm.append(float(line.split()[1]))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("Memory - Performance : gcc O3")    
ax1.set_xlabel('Memory Footprint (kbytes)')
ax1.set_ylabel('Performance (Mflops/s)')
ax1.plot(x_mnk,y_mnk, '-x', c='b', label='mnk data')
ax1.plot(x_mkn,y_mkn, '-x', c='r', label='mkn data')
ax1.plot(x_nmk,y_nmk, '-x', c='g', label='nmk data')
ax1.plot(x_nkm,y_nkm, '-x', c='c', label='nkm data')
ax1.plot(x_kmn,y_kmn, '-x', c='k', label='kmn data')
ax1.plot(x_knm,y_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

x = [5,10,20,50,75,100,200,500,1000,2000]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_title("mkn value - Memory : gcc O3")    
ax1.set_ylabel('Memory Footprint (kbytes)')
ax1.set_xlabel('m/n/k value (int)')
ax1.plot(x,x_mnk, '-x', c='b', label='mnk data')
ax1.plot(x,x_mkn, '-x', c='r', label='mkn data')
ax1.plot(x,x_nmk, '-x', c='g', label='nmk data')
ax1.plot(x,x_nkm, '-x', c='c', label='nkm data')
ax1.plot(x,x_kmn, '-x', c='k', label='kmn data')
ax1.plot(x,x_knm, '-x', c='y', label='knm data')
leg = ax1.legend()
plt.show()

