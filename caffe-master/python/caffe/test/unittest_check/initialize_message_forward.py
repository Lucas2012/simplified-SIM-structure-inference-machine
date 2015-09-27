import numpy

#input shape: scene classes 5 + num of people 14 * action classes 40
lengthx = 5 + 40*14
x_scene = [-5,-4,-3,-2,-1]
x_action = list(xrange(1,40*14+1))
x_unit = x_scene+x_action
l = [];
x = [];
for a in range(0,5):
    x.append(x_unit)
    tmp = [1]*(a+1) + [0]*(14-a-1)
    if a == 4:
        tmp = [1]*14
    l = l + tmp
self_data = x
self_label = l


# design number of people in each frame:
labelst = [1,2,3,4,14]

# generate ground truth for scene scores:
scene_all = []
for batch in range(0,5):
    tmp_scene = labelst[batch]*x_scene + [0]*5*(14-labelst[batch]) + x_scene
    scene_all.append(tmp_scene)

# generage ground truth for all actions in a scene
action_all = []
for batch in range(0,5):
    action_frame = x[batch]
    tmp_action_all = []
    for people in range(0,14):
        print people
        if people < labelst[batch]:
            action_per = action_frame[40*people+5:40*(people+1)+5]
            # replicate corresponding number of messages around each node
            tmp_action = action_per*labelst[batch] + [0]*40*(14-labelst[batch]) + action_per
        else:
            tmp_action = [0]*40*15
        tmp_action_all += tmp_action
    action_all.append(tmp_action_all)
    
# concatenate initialized
gt_position = numpy.append(scene_all,action_all,axis = 1).copy()

# assert equal for this layer's output
for batch1,batch2 in zip(gt_position,gt_position):
    # for every batch
    stop_id = labelst[batch]
    for x,y in zip(batch1,batch2):
        assert(x==y)

