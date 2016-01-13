import unittest
import tempfile
import os
import numpy
import caffe
#import h5py
class simplified_message_out_a2s(caffe.Layer):
    """A layer that take Messages in and output Q (Q=5) type of messages for prediction"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0	
        self.slen = 0
        self.alen = 0
        self.frame_num = 0
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
        # top1: action2scene prediction

        # bottom inputs:
        bottom1_shape = bottom[0].data.shape
        self.bottom_batchsize = [bottom1_shape[0]]
        self.bottom_output_num = [bottom1_shape[1]]
        # label_stop:
        
        labels = bottom[3].data
        self.frame_num = labels.shape[0]/self.nPeople
        label_stop = self.nPeople*numpy.ones([self.frame_num])
        #print self.frame_num
        count = 0
        for i in range(0,self.frame_num):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        top[0].reshape(bottom1_shape[0],2*self.nScene)
        top[1].reshape(bottom1_shape[0],3*self.nAction)
        top[2].reshape(bottom[2].data.shape[0],3*self.nAction)

 
    def forward(self, bottom, top):
        # bottom0: action2scene messages
        # bottom1: scene2action messages
        # bottom2: action2action messages
        # bottom3: labels
        # bottom4: concat_all0(unary)
        # top0: a2s message prediction
        # top1: s2a message prediction
        # top2: a2a message prediction

        label_stop = self.nPeople*numpy.ones([self.frame_num])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.frame_num):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        for i in range(0,self.bottom_batchsize[0]):
            top[0].data[i] = numpy.append(numpy.zeros([1,self.nScene]),bottom[0].data[i]).copy()
            top[1].data[i] = numpy.append(numpy.zeros([1,2*self.nAction]),bottom[1].data[i]).copy()

        '''for i in range(0,bottom[2].data.shape[0]):
            tmp = numpy.append(numpy.zeros([1,self.nAction]),bottom[2].data[i]).copy()
            top[2].data[i] = numpy.append(tmp,numpy.zeros([1,self.nAction])).copy()'''

        idx = 0
        #print self.label_stop
        for i in range(0,bottom[4].data.shape[0]):
            for j in range(0,int(self.label_stop[i])):
                for k in range(0,int(self.label_stop[i]-1)):
                    #tmp = numpy.append(bottom[4].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction],bottom[2].data[idx]).copy()
                    tmp = numpy.append(numpy.zeros(self.nAction),bottom[2].data[idx]).copy()
                    top[2].data[idx] = numpy.append(tmp,numpy.zeros([1,self.nAction])).copy()
                    idx += 1
                    

    def backward(self, top, propagate_down, bottom):
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
