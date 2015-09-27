import numpy
'''
bottom[0] : unary input
bottom[1] : scene->action   [bottom_batchsize*self.nPeople, self.nAction]
bottom[2] : action->scene    [bottom_batchsize*self.nPeople, self.nScene]
bottom[3] : action->action  [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
bottom[4] : action prediction results from former iteration
bottom[5] : frame prediction results from former iteration
(optional)bottom[6] : temporal after->temporal before  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
(optional)bottom[7] : temporal before->temporal after  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
bottom[8] : action labels
'''
# test layer:
# data input: data_x(5,5+14*40), label(70)
# layer bottoms: s2a(5*14,40), a2s(5*14,5), a2a(5*14*14,40)
# tops: message_in(5,8475),message1(5,15*5),message2(5,14,15*40)-->version of no temporal connections
# add the rest input blobs for mesasge_out layer: action_unary(5,560), scene_unary(5,5)

# should all consider dummies??????
# initialize all the input blobs:
x_data_unit = list(xrange(0,565))
x_label = []
#x_action_unary_unit = list(xrange(-560,0))
x_s2a_unit = [1000]*40
x_a2s_unit = [1500]*5 
x_a2a_unit = [2000]*40 
labelst = [1,2,3,4,14]

# action labels in each frame
for a in range(0,5):
    tmp = [1]*(a+1) + [0]*(14-a-1)
    if a == 4:
        tmp = [1]*14
    x_label = x_label + tmp
# input:
x_data = []
x_action_unary = []
x_scene_unary = []
x_s2a = []
x_a2s = []
x_a2a = []
x_data_filtered = []
for i in range(0,5):
    x_data.append(x_data_unit)
    x_data_filtered_unit = list(xrange(0,565))
    x_data_filtered_unit[5+40*labelst[i]:] = (14-labelst[i])*40*[0]
    x_data_filtered.append(x_data_filtered_unit)
    x_action_unary_unit = list(xrange(-565,-565+labelst[i]*40)) + [0]*40*(14-labelst[i])
    x_scene_unary_unit = list(xrange(-565,-560))
    #print len(x_scene_unary_unit)
    x_action_unary.append(x_action_unary_unit)
    x_scene_unary.append(x_scene_unary_unit)
for i in range(0,70):
    x_s2a.append(x_s2a_unit)
    x_a2s.append(x_a2s_unit)
for i in range(0,980):
    x_a2a.append(x_a2a_unit)

# the ground truth positions and values for 3 final output blobs:
y_three = []   # need to filter out dummies
y_message1 = []   # (5,80)
y_message2 = []   # (5,14,640)

s2a = numpy.reshape(x_s2a,[5,14*40]).copy()
a2s = numpy.reshape(x_a2s,[5,14*5]).copy()
a2a = numpy.reshape(x_a2a,[5,14,14*40]).copy()
# filters:
for i in range(0,5):
    idx = labelst[i]
    s2a_unit = s2a[i].copy()
    a2s_unit = a2s[i].copy()
    a2a_unit = a2a[i] .copy()
    x_data_filtered_unit = x_data_filtered[i].copy()
    x_scene_unary_unit = x_scene_unary[i].copy()
    x_action_unary_unit = x_action_unary[i].copy()
    x_action_unary_unit = numpy.reshape(x_action_unary_unit,[14,40]).copy()

    blank = []
    # scene input messages arrangements:
    a2s_unit[5*labelst[i]:] = numpy.zeros([1,(14-labelst[i])*5])
    blank = numpy.append(blank,a2s_unit,axis = 1).copy()
    blank = numpy.append(blank,x_data_filtered_unit[0:5],axis = 1).copy()
    # #blank = numpy.append(blank,x_scene_unary_unit)
    # action input messages arrangements
    a2a_unit[:,labelst[i]*40:] = numpy.zeros([14,(14-labelst[i])*40])  
    a2a_unit[labelst[i]:,:] = numpy.zeros([14-labelst[i],14*40])
    s2a_unit = numpy.reshape(s2a_unit,[14,40])
    s2a_unit[labelst[i]:,:] = numpy.zeros([14-labelst[i],40])
    ## --add unaries for action scores
    x_data_action_tmp = numpy.reshape(x_data_filtered_unit[5:],[14,40]).copy()
    for j in range(0,14):
        a2a_unit[j,40:j*40+40] = a2a_unit[j,0:j*40].copy()
        a2a_unit[j,0:40] = x_data_action_tmp[j].copy()
    a2a_unit = numpy.append(a2a_unit,s2a_unit,axis = 1).copy()
    # #a2a_unit = numpy.append(a2a_unit,x_data_action_tmp, axis = 1)
    tmp = numpy.reshape(a2a_unit,[1,14*15*40]).copy()
    blank = numpy.append(blank,tmp).copy()
    y_three = numpy.append(y_three,blank,axis = 0).copy()

y_three = numpy.reshape(y_three,[5,8475]).copy()
print y_three[3,0:75]
y_message1 = y_three[:,0:75].copy()
y_message2 = numpy.reshape(y_three[:,75:],[5,14,600]).copy()

# input blobs:
self.net.blobs['data'].data[...] = x_data
self.net.blobs['label'].data[...] = x_label
self.net.blobs['s2a'].data[...] = x_s2a
self.net.blobs['a2s'].data[...] = x_a2s
self.net.blobs['a2a'].data[...] = x_a2a
self.net.blobs['action_unary'].data[...] = x_action_unary
self.net_blobs['scene_unary'].data[...] = x_scene_unary

# check output:
self.net.forward()
for batch1,batch2 in zip(self.net.blobs['three'].data,y_three):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['message1'].data,y_message1):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['message2'].data,y_message2):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

