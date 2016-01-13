import unittest
import tempfile
import os
import numpy
import caffe
#import h5py
class simplified_message_in(caffe.Layer):
    """A layer that take Messages in and output Q (Q=5) type of messages for prediction"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0
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
        for i in range(0,len(self.top_batchsize)):
            top[i].reshape(max(1,self.top_batchsize[i]),max(1,self.top_output_num[i]))
            
 
    def forward(self, bottom, top):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction 
        
        #print 'forward',bottom[0].data[0]
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[1].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
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
                        top[potential_type].data[int(count[potential_type])] = bottom[0].data[i,self.nScene+(m-1)*self.nAction:self.nScene+m*self.nAction]
                    count[potential_type] += 1      
        #print 'in',top[1].data  
        #print  'label_frame',bottom[2].data  
        #print 'a2s_input',top[1].data      
        

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

        #print 'in',top[1].data  
        #print  'label_frame',bottom[2].data
        #print '********    '

def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 10 dim: 9 dim: 8 }
        layer { type: 'Python' name: 'one' bottom: 'data' top: 'one'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'two' bottom: 'one' top: 'two'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }
        layer { type: 'Python' name: 'three' bottom: 'two' top: 'three'
          python_param { module: 'test_python_layer' layer: 'SimpleLayer' } }""")
        return f.name

class TestPythonLayer(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        x = 8
        self.net.blobs['data'].data[...] = x
        self.net.forward()
        for y in self.net.blobs['three'].data.flat:
            self.assertEqual(y, 10**3 * x)

    def test_backward(self):
        x = 7
        self.net.blobs['three'].diff[...] = x
        self.net.backward()
        for y in self.net.blobs['data'].diff.flat:
            self.assertEqual(y, 10**3 * x)

    def test_reshape(self):
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
