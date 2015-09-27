import numpy

# label stop:
labelst = [1,2,3,4,14]

# top diffs:  length 7910 = 5*(14+1) + 14*(14+1)*40 = 75 + 8400
x_unit = list(xrange(0,8475))
x = []
for i in range(0,5):
    x.append(x_unit)

# generate ground truth scene diffs:
scene_diff = []
for i in range(0,5):
    frame_scene = x[i]    
    scene_diff.append(frame_scene[70:75])        

# generate ground truth action diffs:
action_diff = []
for i in range(0,5):
    per_frame = []
    for j in range(0,14):
        if j < labelst[i]:
            action_x = x[i]
            action_unary = action_x[j*40+75+j*600:(j+1)*40+75+j*600]
        else:
            action_unary = [0]*40    
        per_frame += action_unary
    action_diff.append(per_frame)

# concatenate scene and action
gt_position = numpy.append(scene_diff,action_diff,axis = 1).copy()

# the bottom diff should equal to specific positions

for batch1,batch2 in zip(gt_position,gt_position):
    print 'batch length:',len(batch1)
    print batch1
    for x,y in zip(batch1,batch2):
        assert(y, x)
