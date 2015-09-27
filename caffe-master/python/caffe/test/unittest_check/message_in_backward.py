import numpy
# 2 bottoms: message_in(8475), label_action(70); 
# 3/5 tops: s2a(5*14,5), a2s(5*14,40), a2a(5*14*14,40)

# label values:
labelst = [1,2,3,4,14]

# assign all values for top diffs to be 1:
s2a_diff = numpy.ones([70,5])
a2s_diff = numpy.ones([70,40])
a2a_diff = numpy.ones([980,40])

# ground truth diff values for original input blobs:
# the diff for input blobs should all be unary diffs, and should all be 1.0:
scene_node_diff = numpy.ones([5,5])
action_node_diff = numpy.ones([5,14*40])
for i in range(0,5):
    for j in range(0,14):
        action_node_diff[i,labelst[i]*40:] = numpy.zeros([1,40*(14-labelst[i])])
data_diff = numpy.append(scene_node_diff,action_node_diff,axis = 1).copy()


# top diff values: 
self.net.blobs['s2a'].diff[...] = s2a_diff.copy()
self.net.blobs['a2s'].diff[...] = a2s_diff.copy()
self.net.blobs['a2a'].diff[...] = a2a_diff.copy()
self.net.backward()
for batch1,batch2 in zip(self.net.blobs['data'].diff,data_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

