import unittest
import tempfile
import os
import numpy
import caffe

class test_message_in(caffe.Layer):
    """A layer that take Messages in and output Q (Q=5) type of messages for prediction"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 40
        self.nPeople = 14
        self.K_ = 0	;
        self.slen = 0
        self.alen = 0
        self.tlen_leaf = 0
        self.tlen_mid = 0 
        self.aunit = 0
        self.message_num_action = self.nPeople+2*(self.K_>0)+1
        self.label_stop = []

        # controls:
        self.ifnormalize = True
        self.ifprevious = False
        self.ifensemble = True
        self.id = 0
        self.prevhardgate = True

        self.graph_structure=[]
        
        # shapes
        self.bottom_batchsize = []
        self.top_batchsize = []
        self.bottom_output_num = []
        self.top_output_num = []
        self.id = 0
    
    def reshape(self, bottom, top):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction 

        # bottom inputs:
        self.id += 1

        bottom1_shape = bottom[0].data.shape
        self.bottom_batchsize = [bottom1_shape[0]]
        self.bottom_output_num = [bottom1_shape[1]]

        # label_stop:
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[1].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop

        # A general map for graph structure: [num_node, num_node], val(x,y) is the type of potential function
        # node order:[scene, (current frame) p1, p2, ..., pn, (temporal frames), pn+1,...]
        # now only current frame
        # a2a:
        graph_structure = numpy.zeros([self.nPeople+1,self.nPeople+1])
        # a2s:
        graph_structure[:,0] = 1
        # s2a:
        graph_structure[0,:] = 2
        # clear self loop
        for i in range(0,self.nPeople+1):
            graph_structure[i,i] = -1
        self.graph_structure = numpy.zeros([len(graph_structure),len(graph_structure),self.bottom_batchsize[0]])

        # (optional) change graph structures for each frame
        for i in range(0,self.bottom_batchsize[0]):
            temp_graph = graph_structure.copy()
            num_p = label_stop[i]
            temp_graph[num_p+1:,:] = -1
            temp_graph[:,num_p+1:] = -1
            self.graph_structure[:,:,i] = temp_graph.copy()

        # top outputs:
        a2a_num = numpy.sum(self.graph_structure==0)
        a2s_num = numpy.sum(self.graph_structure==1)
        s2a_num = numpy.sum(self.graph_structure==2)

        self.top_batchsize = [a2a_num,a2s_num,s2a_num]
        self.top_output_num = [self.nAction, self.nAction, self.nScene]
        #print self.label_stop
        for i in range(0,len(self.top_batchsize)):
            #print 'top shape',self.top_batchsize[i],self.top_output_num[i]
            top[i].reshape(max(1,self.top_batchsize[i]),max(1,self.top_output_num[i]))
            #print top[i].data.shape
            #top[i].reshape(self.id,4)
            
 
    def forward(self, bottom, top):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction 
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[1].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]        
        self.label_stop = label_stop

        count = numpy.zeros(10)
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,len(self.graph_structure)):
                for m in range(0,len(self.graph_structure)):
                    potential_type = int(self.graph_structure[m,j,i])
                    if potential_type == -1:
                        continue
                    if m == 0:
                        top[potential_type].data[int(count[potential_type])] = bottom[0].data[i,:self.nScene].copy()
                    else:
                        #print count[potential_type]
                        top[potential_type].data[int(count[potential_type])] = bottom[0].data[i,self.nScene+(m-1)*self.nAction:self.nScene+m*self.nAction]
                    count[potential_type] += 1                
        

    def backward(self, top, propagate_down, bottom):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction         

        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[1].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop
        count = numpy.zeros(10)
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,len(self.graph_structure)):
                for m in range(0,len(self.graph_structure)):
                    potential_type = int(self.graph_structure[m,j,i])
                    if potential_type == -1:
                        continue
                    if m == 0:
                        bottom[0].diff[i,:self.nScene] += top[potential_type].diff[count[potential_type]].copy()
                    else:
                        bottom[0].diff[i,self.nScene+(m-1)*self.nAction:self.nScene+m*self.nAction] += top[potential_type].diff[count[potential_type]].copy()



def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 5 dim: 565 }
        input: 'label' input_shape {dim: 70}
        layer { type: 'Python' name: 'one' bottom: 'data' bottom: 'label' top: 'a2a' top: 'a2s' top: 's2a'
          python_param { module: 'test_simplified_message_in' layer: 'test_message_in' } }""")
        return f.name

class TestMessageIn(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
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
                    continue
                else:
                    s2a_gt.append([-1]*5)
        
        # gt_position for a2s messages
        a2s_gt = []
        for f in range(0,5):
            for p in range(0,14):
                if p+1 > labelst[f]:
                    continue
                else:
                    a2s_gt.append([p+1]*40)                        

        # gt_position for a2a messages
        a2a_gt = []
        for f in range(0,5):
            per_frame = []
            for p1 in range(0,14):
                for p2 in range(0,14):
                    if p2+1 > labelst[f] or p1+1 > labelst[f] or p1 == p2:
                        #if p1 == p2:
                        #    a2a_gt.append([0]*40)
                        continue
                    else:
                        a2a_gt.append([p2+1] *40)

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
        #print 'len',len(a2a),len(a2a_gt)
        for batch in range(0,len(a2a_gt)):
            #print a2a[batch]
            #print a2a_gt[batch]
            #print batch
            for x,y in zip(a2a[batch],a2a_gt[batch]):
                self.assertEqual(x,y)

    def test_backward(self):
        # 2 bottoms: message_in(8475), label_action(70); 
        # 3/5 tops: s2a(5*14,5), a2s(5*14,40), a2a(5*14*14,40)

        # label values:
        labelst = [1,2,3,4,14]
        l = [];
        for a in range(0,5):
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            l = l + tmp
        self.net.blobs['label'].data[...] = l

        # assign all values for top diffs to be 1:
        s2a_diff = numpy.ones([24,5])
        a2s_diff = numpy.ones([24,40])
        a2a_diff = numpy.ones([202,40])
 
        # ground truth diff values for original input blobs:
        # the diff for input blobs should all be unary diffs, and should all be 1.0:
        scene_node_diff = numpy.ones([5,5])
        action_node_diff = numpy.ones([5,14*40])
        for i in range(0,5):
            scene_node_diff[i] *= labelst[i]
            for j in range(0,14):
                if j >=labelst[i]:
                    action_node_diff[i,j*40:(j+1)*40] = numpy.zeros(40)
                    continue
                action_node_diff[i,j*40:(j+1)*40]  *= labelst[i]
        data_diff = numpy.append(scene_node_diff,action_node_diff,axis = 1)

        # top diff values: 
        self.net.blobs['s2a'].diff[...] = s2a_diff
        self.net.blobs['a2s'].diff[...] = a2s_diff
        self.net.blobs['a2a'].diff[...] = a2a_diff
        self.net.backward()
        pos = 0
        for batch1,batch2 in zip(self.net.blobs['data'].diff,data_diff):
            #print "message in"
            #print batch1
            for x,y in zip(batch1,batch2):
                #print pos
                pos += 1
                self.assertEqual(y, x)

    def test_reshape(self):
        # do nothing for reshape layer test
        return
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
