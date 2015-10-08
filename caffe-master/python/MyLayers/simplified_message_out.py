import unittest
import tempfile
import os
import numpy
import caffe
#import h5py
class simplified_message_out(caffe.Layer):
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
    
    def reshape(self, bottom, top):
        # bottom1: message in
        # bottom2: label_action
        # top1: action2action prediction
        # top2: action2scene prediction
        # top3: scene2action prediction 

        # bottom inputs:
        bottom1_shape = bottom[4].data.shape
        self.bottom_batchsize = [bottom1_shape[0]]
        self.bottom_output_num = [bottom1_shape[1]]
        # label_stop:
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
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
        self.top_batchsize = [self.bottom_batchsize[0],self.bottom_batchsize[0]*self.nPeople]
        self.top_output_num = [self.nScene*2, self.nAction*3]
        for i in range(0,len(self.top_batchsize)):
            top[i].reshape(self.top_batchsize[i],self.top_output_num[i])

 
    def forward(self, bottom, top):
        # bottom0: action2action prediction
        # bottom1: action2scene prediction
        # bottom2: scene2action prediction 
        # bottom3: labels
        # bottom4: unaries

        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
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
                    unit[(pos-1)*unit_len:pos*unit_len] += bottom[potential_type].data[count[potential_type]]
                    count[potential_type] += 1
                for k in range(1,len(potentials)):
                    if num_po[k] != 0:
                        unit[unit_len*(k-1):unit_len*k] /= num_po[k]
                if j == 0:
                    unit = numpy.append(unary[i,:self.nScene],unit)
                    top[0].data[int(top_count[0])] = unit
                    top_count[0] += 1
                else:
                    unit = numpy.append(unary[i,self.nScene+(j-1)*self.nAction:self.nScene+j*self.nAction],unit).copy()
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
                    if numpy.sum(self.graph_structure[:,j,i] == -1) != 0:
                        bottom[4].diff[i,self.nScene+(j-1)*self.nAction:self.nScene+j*self.nAction] = unary_diff.copy()
                if numpy.sum(self.graph_structure[:,j,i] == -1) == 0:
                    continue;
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
                    bottom[potential_type].diff[count[potential_type]] = unit[pos*unit_len:(pos+1)*unit_len]/num_po[pos].copy()
                    count[potential_type] += 1

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
