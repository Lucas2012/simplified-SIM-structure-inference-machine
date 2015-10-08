import unittest
import tempfile
import os
import numpy
import caffe
#import h5py
class simplified_message_out_cavity(caffe.Layer):
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
        self.function_type = 0

        self.graph_structure=[]
        
        # shapes
        self.bottom_batchsize = []
        self.top_batchsize = []
        self.bottom_output_num = []
        self.top_output_num = []
    
    def reshape(self, bottom, top):
        # bottom0: gate_input
        # bottom1: cavity_prediction
        # bottom2: a2s_message
        # bottom3: label_action
        # bottom4: diff_receiver
        # top1: entropy/KL loss
        

        # bottom inputs:
        bottom1_shape = bottom[2].data.shape
        self.bottom_batchsize = [bottom1_shape[0]/self.nPeople]
        #self.bottom_output_num = [bottom1_shape[1]]
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
        top[i].reshape(1)

 
    def forward(self, bottom, top):
        # bottom0: gate_input
        # bottom1: cavity_prediction
        # bottom2: a2s_message
        # bottom3: label_action
        # bottom4: diff_receiver
        # top1: entropy/KL loss

        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize[0]])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        step = 0     
        count = 0   
        loss_all = 0.0
        for f in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.label_stop[f]): 
                o = bottom[0].data[count].copy()
                cavity_prediction = bottom[1].data[count].copy()
                a2s_message = bottom[2].data[count].copy()
                loss = 0
                distance_function(loss,cavity_prediction,a2s_message,self.function_type)
                loss_all += ((1+numpy.tanh(o))/2.0)*loss
                count += 1
 
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
 
        step = 0
        count = 0
        for f in range(0,self.bottom_batchsize[0]):
            for j in range(0,self.label_stop[f]):
                # predicted message diff:
                o = bottom[0].data[count].copy()
                cavity_prediction = bottom[1].data[count].copy()
                a2s_message = bottom[2].data[count].copy() 
                diff = numpy.zeros(len(bottom[1].data[0]))
                function_diff(diff,cavity_prediction,a2s_message,self.function_type)
                bottom[1].diff[count] = diff.copy()

                # diff receiver: to do
                loss = 0.0
                distance_function(loss,cavity_prediction,a2s_message,self.function_type)
                bottom[4].diff[count] = loss;
                count += 1

    def distance_function(loss,p,q,function_type):
        # KL divergence
        if function_type == 0:
            loss = 0.0
            for i in range(len(p)):
                loss += p[i]*numpy.log(p[i]/q[i])
        # cross entropy
        elif function_type == 1:
            loss = 0.0 
            for i in range(len(p)):
                loss -= p[i]*numpy.log(q[i])   

    def function_diff(diff,p,q,function_type):
        # KL divergence
        if function_type == 0:
            diff = 0.0*diff
            for i in range(0,len(p)):
                diff[i] = numpy.log(p[i]/q[i]) + 1
        # cross entropy: to be done
        elif function_type == 1:
            pass
           

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
