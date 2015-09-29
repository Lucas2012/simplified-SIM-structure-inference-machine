import unittest
import tempfile
import os
import numpy
import caffe

class test_message_out(caffe.Layer):
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
    
    def reshape(self, bottom, top):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction 

        # bottom inputs:
        print 'check p1'
        bottom1_shape = bottom[4].data.shape
        print bottom1_shape[0]
        self.bottom_batchsize = [bottom1_shape[0]]
        self.bottom_output_num = [bottom1_shape[1]]
        print 'check p2'
        # label_stop:
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop
        print 'check p3'
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
        print 'check p4'
        # (optional) change graph structures for each frame
        for i in range(0,self.bottom_batchsize[0]):
            temp_graph = graph_structure.copy()
            num_p = label_stop[i]
            temp_graph[num_p+1:,:] = -1
            temp_graph[:,num_p+1:] = -1
            self.graph_structure[:,:,i] = temp_graph.copy()
        print 'check p5'
        # top outputs:
        
        self.top_batchsize = [self.bottom_batchsize[0],self.bottom_batchsize[0]*self.nPeople]
        self.top_output_num = [self.nScene*2, self.nAction*3]
        for i in range(0,len(self.top_batchsize)):
            top[i].reshape(self.top_batchsize[i],self.top_output_num[i])
        print 'check p6' 

 
    def forward(self, bottom, top):
        # bottom0: action2action prediction
        # bottom1: action2scene prediction
        # bottom2: scene2action prediction 
        # bottom3: labels
        # bottom4: unaries
        print 'forward'
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop

        unary = bottom[4].data

        count = numpy.zeros(10)
        top_count = numpy.zeros(2)
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,len(self.graph_structure)):
                if j == 0:
                    unit = numpy.zeros(self.nScene)
                    unit_len = self.nScene
                else:
                    unit = numpy.zeros(2*self.nAction)
                    unit_len = self.nAction
                potentials = numpy.unique(self.graph_structure[:,j,i])
                if j != 0 and numpy.any(potentials==0) == False:
                    potentials = numpy.append(potentials,0)
                potentials = numpy.sort(potentials)
                num_po = numpy.zeros(len(potentials))
                for k in range(0,len(potentials)):
                    num_po[k] = numpy.sum(potentials[k] == self.graph_structure[:,j,i])
                inner_count = numpy.zeros(3)
                for m in range(0,len(self.graph_structure)):
                    potential_type = int(self.graph_structure[m,j,i])
                    if potential_type == -1:
                        continue
                    pos = numpy.where(potential_type == potentials)
                    pos = pos[0][0]
                    #print potentials
                    #print 'type',potential_type
                    unit[(pos-1)*unit_len:pos*unit_len] += bottom[potential_type].data[count[potential_type]]
                    count[potential_type] += 1
                    assert(unit[0] >= 0)
                for k in range(1,len(potentials)):
                    if num_po[k] != 0:
                        unit[unit_len*(k-1):unit_len*k] /= num_po[k]
                if j == 0:
                    unit = numpy.append(unary[i,:self.nScene],unit)
                    #print top_count[0]
                    #print len(unit)
                    #print unary
                    top[0].data[int(top_count[0])] = unit
                    top_count[0] += 1
                else:
                    unit = numpy.append(unary[i,self.nScene+(j-1)*self.nAction:self.nScene+j*self.nAction],unit)
                    top[1].data[top_count[1]] = unit.copy()
                    top_count[1] += 1


    def backward(self, top, propagate_down, bottom):
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop
        count = numpy.zeros(10)
        top_count = numpy.zeros(2)
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,len(self.graph_structure)):
                if j == 0:
                    unit = top[0].diff[top_count[0]].copy()
                    top_count[0] += 1
                    unit_len = self.nScene
                    unary_diff = unit[:unit_len].copy()
                    bottom[4].diff[i,:self.nScene] = unary_diff.copy()
                else:
                    unit = top[1].diff[top_count[1]].copy()
                    top_count[1] += 1
                    unit_len = self.nAction
                    unary_diff = unit[:unit_len].copy()
                    bottom[4].diff[i,self.nScene+(j-1)*self.nAction:self.nScene+j*self.nAction] = unary_diff.copy()
                potentials = numpy.unique(self.graph_structure[:,j,i])
                if j != 0 and numpy.any(potentials==0) == False:
                    potentials = numpy.append(potentials,0)
                potentials = numpy.sort(potentials)
                num_po = numpy.zeros(len(potentials))
                for k in range(0,len(potentials)):
                    num_po[k] = numpy.sum(potentials[k] == self.graph_structure[:,j,i])
                for m in range(0,len(self.graph_structure)):
                    potential_type = int(self.graph_structure[m,j,i])
                    if potential_type == -1:
                        continue
                    pos = numpy.where(potential_type == potentials)
                    pos = pos[0][0]
                    #print potential_type
                    
                    bottom[potential_type].diff[count[potential_type]] = unit[pos*unit_len:(pos+1)*unit_len]/num_po[pos].copy()
                    count[potential_type] += 1
             

def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'a2a' input_shape { dim: 202 dim: 40 }
        input: 'a2s' input_shape { dim: 24 dim: 5 }
        input: 's2a' input_shape { dim: 24 dim: 40 }
        input: 'label' input_shape {dim: 70}
        input: 'unaries' input_shape { dim: 5 dim: 565 }
        layer { type: 'Python' name: 'one' bottom: 'a2a' bottom: 'a2s' bottom: 's2a' bottom: 'label' bottom: 'unaries' top: 'scene' top: 'action'
          python_param { module: 'test_simplified_message_out' layer: 'test_message_out' } }""")
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
        x_unit = numpy.array(x_unit)
        x = numpy.zeros([5,565])
        l = []
        # action labels in each frame
        for a in range(0,5):
            x[a] = x_unit.copy()
            x[a,5+labelst[a]*40:] = numpy.zeros(40*(14-labelst[a]))
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            l = l + tmp

        # s2a messages
        s2a = []
        for a in range(0,5):
            for p in range(0,14):
                if p+1 > labelst[a]:
                    continue
                else:
                    s2a.append([-1]*40)
        
        # a2s messages
        a2s = []
        for f in range(0,5):
            for p in range(0,14):
                if p+1 > labelst[f]:
                    continue
                else:
                    a2s.append([p+1]*5)                        

        # a2a messages
        a2a = []
        for f in range(0,5):
            per_frame = []
            for p1 in range(0,14):
                for p2 in range(0,14):
                    if p2+1 > labelst[f] or p1+1 > labelst[f] or p1 == p2:
                        continue
                    else:
                        a2a.append([p2+1] *40)

        # gt scenes:
        scene_gt = []
        for f in range(0,5):
            unary = [-1]*5
            ls = labelst[f]
            pairwise = [ls*(ls+1)/2.0/ls]*5
            #print 'ls',ls*(ls+1)/2.0/ls
            unit = unary+pairwise
            scene_gt.append(unit)
        # gt actions:
        action_gt = []
        for f in range(0,5):
            ls = labelst[f]
            for p in range(0,14):
                if p+1 > labelst[f]:
                    action_gt.append([0]*120)
                    continue
                unary = [p+1]*40
                pairwise = [max(0,(ls+1)*ls/2-p-1)/(1.0*max(1,ls-1))]*40
                third = [-1]*40
                unit = unary+pairwise+third
                action_gt.append(unit)
        self.net.blobs['s2a'].data[...] = s2a
        self.net.blobs['a2s'].data[...] = a2s
        self.net.blobs['a2a'].data[...] = a2a
        #print 'check data',a2a
        self.net.blobs['label'].data[...] = l
        self.net.blobs['unaries'].data[...] = x
        self.net.forward()
        scene = self.net.blobs['scene'].data
        action = self.net.blobs['action'].data
        #print 'data',a2s
        for batch in range(0,len(scene_gt)):
            for x,y in zip(scene[batch],scene_gt[batch]):
                self.assertEqual(x,y)

        for batch in range(0,len(action_gt)):
            for x,y in zip(action[batch],action_gt[batch]):
                pass#self.assertEqual(x,y)

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

        # scene node diff:
        scene_unit = numpy.ones(10)
        scene_diff_gt = numpy.zeros([5,10])
        for f in range(0,5):
            ls = labelst[f]
            tmp = scene_unit.copy()
            tmp[5:] *= ls
            scene_diff_gt[f] = tmp.copy()
  
        # action node diff:
        action_unit = numpy.ones(120)
        action_diff_gt = numpy.zeros([70,120])
        count_a = 0
        for f in range(0,5):
            ls = labelst[f]
            for p in range(0,14):
                if p+1>labelst[f]:
                    count_a += 1
                    continue
                tmp = action_unit.copy()
                tmp[:40] *= p+1
                tmp[40:80]*=ls-1
                tmp[80:]*=1
                action_diff_gt[count_a] = tmp.copy()
                #print action_diff_gt[count_a]
                count_a += 1

        # top diff values: 
        self.net.blobs['scene'].diff[...] = scene_diff_gt
        self.net.blobs['action'].diff[...] = action_diff_gt
        self.net.backward()
        pos = 0
        for batch1 in self.net.blobs['a2a'].diff:
            #print 'batch',batch1
            pos += 1
            #print pos
            for y in batch1:
                #pos += 1
                pass#self.assertEqual(y, 1.0)
        for batch1 in self.net.blobs['a2s'].diff:
            for y in batch1:
                #print pos
                pos += 1
                self.assertEqual(y, 1.0)
        for batch1 in self.net.blobs['s2a'].diff:
            for y in batch1:
                #print pos
                pos += 1
                self.assertEqual(y, 1.0)

    def test_reshape(self):
        # do nothing for reshape layer test
        return
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
