import sys
import bisect
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 25,
        }
matplotlib.rcParams.update({'font.size': 22})

#sampling = 40
y = [ int(line.strip()) for line in open(sys.argv[1]).readlines() ][:1000]
x = range(1,len(y)+1)

x = np.asarray(x)

plt.subplot(411) # random
plt.xlim(-10,1010)
plt.text(3, 2200, 'Random', fontdict=font)
plt.ylabel('Utterance length')
plt.xlabel('Utterance order')
yr = np.asarray(y)

sc = plt.scatter(x,yr, label='random')


plt.subplot(412) # sorted
plt.xlim(-10,1010)
plt.text(3, 2200, 'Sorted', fontdict=font)

ys = sorted(y)
ys = np.asarray(ys)

plt.scatter(x,ys)
plt.ylabel('Utterance length')
plt.xlabel('Utterance order')

plt.subplot(413) # bucket
plt.xlim(-10,1010)
plt.text(3, 2200, 'Bucketing', fontdict=font)

buckets = [0] + range(250,max(y),250) + [max(y)]
bucket_map = {}
for l in y:
  b = bisect.bisect_left(buckets, l)
  if not buckets[b] in bucket_map:
    bucket_map[buckets[b]] = []
  bucket_map[buckets[b]].append(l)
  
yb = []
for k in buckets:
  if k in bucket_map.keys():
    yb += bucket_map[k]

plt.scatter(x,yb)
plt.ylabel('Utterance length')
plt.xlabel('Utterance order')

plt.subplot(414) # laplace
plt.xlim(-10,1010)
plt.text(3, 2200, 'Proposed approach', fontdict=font)

N = 12
start = 0
end = len(y) / N
yl = []
i = 0
while start < len(y):
  i += 1
  if i % 2 == 1:
    yl += sorted(y[start:end])
  else:
    yl += sorted(y[start:end])[::-1]
  start = end
  end += len(y) / N
  end = min(len(y),end)

plt.scatter(x,yl)
plt.ylabel('Utterance length')
plt.xlabel('Utterance order')


plt.subplots_adjust(hspace=0.33)
plt.show()
