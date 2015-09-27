import numpy
# tops: message_in(5,8475), message1(5,15*5), message2(5,14,15*40)
# bottoms: unary_data(5,565), s2a(5*14,40), a2s(5*14,40), a2a(14*14*5,40), action_pred(5,560), scene_pred(5,5)

# mistake one: fixed unary is also set for "message_in" output, which is incorrect
# warning one: previous prediction scores not used

labelst = [1,2,3,4,14]

messagein_diff_unit = numpy.reshape(list(xrange(0,8475)),[8475])
message1_diff_unit = numpy.reshape(list(xrange(10000,10075)),[75])
message2_diff_unit = numpy.reshape(list(xrange(-14*15*40,0)),[14,600])
messagein_diff = []
message1_diff = []
message2_diff = numpy.zeros([0,600])

for i in range(0,5):
    messagein_diff = numpy.append(messagein_diff,messagein_diff_unit,axis = 0).copy()
    message1_diff = numpy.append(message1_diff,message1_diff_unit,axis = 0).copy()
    message2_diff = numpy.append(message2_diff,message2_diff_unit,axis = 0).copy()
message1_diff = numpy.reshape(message1_diff,[5,75]).copy()
message2_diff = numpy.reshape(message2_diff,[5,14,600]).copy()
# change fixed unaries:
for i in range(0,5):
    for j in range(0,14):
        unary_score = message2_diff[i,j,0:40].copy()
        message2_diff[i,j,0:j*40] = message2_diff[i,j,40:(j+1)*40].copy()
        message2_diff[i,j,j*40:(j+1)*40] = unary_score.copy()
messagein_diff = numpy.reshape(messagein_diff,[5,8475]).copy()
tmp_action_diff = numpy.reshape(message2_diff,[5,8400]).copy()
assert(tmp_action_diff[2,0] == -8400)
assert(tmp_action_diff[2,8399] == -1)
messageall_diff = numpy.append(message1_diff,tmp_action_diff,axis = 1).copy()
messageall_diff += messagein_diff

# unary scene diff:
unary_scene_diff = messageall_diff[:,70:75].copy()

# unary action diff:
print messageall_diff.shape
tmp = numpy.reshape(messageall_diff[:,75:],[5,14,600]).copy()
unary_action_diff = tmp[:,:,0:40].copy()
unary_action_diff = numpy.reshape(unary_action_diff,[5,560]).copy()
for i in range(0,5):
    for j in range(0,14):
        if j >= labelst[i]:
            unary_action_diff[i,j*40:(j+1)*40] = numpy.zeros(40)
        else:
            unary_action_diff[i,j*40:(j+1)*40] = tmp[i,j,j*40:(j+1)*40]
# unary data diff:
unary_data_diff = numpy.append(unary_scene_diff,unary_action_diff,axis = 1).copy()

# s2a diff:
tmp = numpy.reshape(messageall_diff[:,75:],[70,600]).copy()
s2a_diff = tmp[:,560:600].copy()
s2a_diff = numpy.reshape(s2a_diff,[5,560])
for i in range(0,5):
    s2a_diff[i,labelst[i]*40:] = numpy.zeros((14-labelst[i])*40)
s2a_diff = numpy.reshape(s2a_diff,[70,40])

# a2s diff:
a2s_diff = numpy.reshape(messageall_diff[:,0:70],[70,5]).copy()
a2s_diff = numpy.reshape(a2s_diff,[5,70])
for i in range(0,5):
    a2s_diff[i,labelst[i]*5:] = numpy.zeros((14-labelst[i])*5)

# a2a diff:          
tmp = numpy.reshape(messageall_diff[:,75:],[5,14,600]).copy()
tmp = tmp[:,:,:560].copy()
for i in range(0,5):
    for j in range(0,14):
        tmp[i,j,labelst[i]*40:] = numpy.zeros((14-labelst[i])*40)
        tmp[i,j,j*40:(j+1)*40] = numpy.zeros(40)
a2a_diff = numpy.reshape(tmp,[980,40]).copy()

# diff of predictions should all be zeros
action_unary_diff = numpy.zeros([5,560])
scene_unary_diff = numpy.zeros([5,5])

self.net.blobs['three'].diff[...] = messagein_diff.copy()
self.net.blobs['message1'].diff[...] = message1_diff.copy()
self.net.blobs['message2'].diff[...] = message2_diff.copy()
self.net.backward()

# check: 'data','s2a,'a2s','a2a','action_unary','scene_unary'
for batch1,batch2 in zip(self.net.blobs['data'].diff,unary_data_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['s2a'].diff,s2a_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['a2s'].diff,a2s_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['a2a'].diff,a2a_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['action_unary'].diff,action_unary_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

for batch1,batch2 in zip(self.net.blobs['scene_unary'].diff,scene_unary_diff):
    for x,y in zip(batch1,batch2):
        self.assertEqual(y, x)

