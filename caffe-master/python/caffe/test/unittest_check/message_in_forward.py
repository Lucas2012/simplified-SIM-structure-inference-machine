# 2 bottoms, 3/5 tops: message_in(8475), label_action(70); s2a,a2s,a2a

# design number of people in each frame:
labelst = [1,2,3,4,14]

# for each frame:
x_unit = [-1]*5
for p in range(0,14):
    x_unit += [p+1]*40
x = []
l = []
# action labels in each frame
for a in range(0,5):
    x.append(x_unit)
    tmp = [1]*(a+1) + [0]*(14-a-1)
    if a == 4:
        tmp = [1]*14
    l = l + tmp

# gt_position for s2a messages
s2a_gt = []
for a in range(0,5):
    for p in range(0,14):
        if p+1 > labelst[a]:
            s2a_gt.append([0]*5)
        else:
            s2a_gt.append([-1]*5)

# gt_position for a2s messages
a2s_gt = []
for f in range(0,5):
    for p in range(0,14):
        if p+1 > labelst[f]:
            a2s_gt.append([0]*40)
        else:
            a2s_gt.append([p+1]*40)                        

# gt_position for a2a messages
a2a_gt = []
for f in range(0,5):
    per_frame = []
    for p1 in range(0,14):
        for p2 in range(0,14):
            if p2+1 > labelst[f] or p1+1 > labelst[f]:
                a2a_gt.append([0]*40)
            else:
                a2a_gt.append([p1+1] *40)
print len(a2a_gt)
print l
self.net.blobs['data'].data[...] = x
self.net.blobs['label'].data[...] = l
self.net.forward()
s2a = self.net.blobs['s2a'].data
a2s = self.net.blobs['a2s'].data
a2a = self.net.blobs['a2a'].data
for batch in range(0,len(s2a_gt)):
    for x,y in zip(s2a[batch],s2a_gt[batch]):
        self.assertEqual(x,y)
    for x,y in zip(a2s[batch],a2s_gt[batch]):
        self.assertEqual(x,y)
    for x,y in zip(a2a[batch],a2a_gt[batch]):
        self.assertEqual(x,y)

