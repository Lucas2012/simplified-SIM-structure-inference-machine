import unittest
import tempfile
import os
import numpy

import caffe

class Data_Arrange_Layer(caffe.Layer):
    """A layer that initialize messages for recurrent belief propagation"""

    def setup(self, bottom, top):
        self.nScene = 5
        #self.nAction = 40
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0;
        self.bottom_batchsize = 0
        self.unitlen = 0
        self.output_num = 0
        self.bottom_batchsize = 0
        self.top_batchsize = 0
        self.bottom_output_num = 0
        self.top_output_num = 0
        self.top_shape = []
        self.T_ = 1
        self.ifaverage = False
    
    def reshape(self, bottom, top):
        # have one input one output, initialize messages for each node in the graphical model 
        bottom_shape = bottom[0].data.shape
        self.bottom_batchsize = bottom_shape[0]/self.T_
        self.top_batchsize = self.bottom_batchsize
        self.bottom_output_num = bottom_shape[1]
        top[0].reshape(self.top_batchsize, self.bottom_output_num)
        #print self.bottom_batchsize
        

    def forward(self, bottom, top):
        #print self.T_
        #print numpy.argmax(bottom[0].data)
        print bottom[1].data[0]
        if self.ifaverage:
            bottom_data = bottom[0].data[:self.bottom_batchsize].copy()
            for i in range(1,self.T_):
                bottom_data += bottom[0].data[i*self.bottom_batchsize:(i+1)*self.bottom_batchsize].copy() 
            top[0].data[...] = bottom_data.copy()/self.T_
        else:
            top[0].data[...] = bottom[0].data[(self.T_-1)*self.bottom_batchsize:self.T_*self.bottom_batchsize].copy()      
            #top[0].data[...] = bottom[0].data[(0)*self.bottom_batchsize:1*self.bottom_batchsize].copy()      
              

    def backward(self, top, propagate_down, bottom):
        a = bottom[10].data
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
