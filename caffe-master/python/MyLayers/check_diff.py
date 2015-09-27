import unittest
import tempfile
import os
import numpy

import caffe

class check_diff(caffe.Layer):
    """A layer that initialize messages for recurrent belief propagation"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 40
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
        self.label_stop = []
    
    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        top[0].data[...] = bottom[0].data
        
        '''if len(top[0].data) <= 5:
            return
        print "output for all iters:"
        print "check all data:"
        print top[0].data.shape
        print "first data:"
        print top[0].data[0:5]
        print "second data:"
        print top[0].data[5:10]
        print "third data:"
        print top[0].data[10:15]
        print "fourth data:"
        print top[0].data[15:20]'''
        #print 'context'
        
    def backward(self, top, propagate_down, bottom):
        bottom[0].diff[...] = top[0].diff
        #print bottom[0].diff
        '''if len(bottom) > 1:
            print "check all diff:"
            print top[0].diff.shape
            print "first diff:"
            print top[0].diff[0:5]
            print "second diff:"
            print top[0].diff[5:10]
            print "third diff:"
            print top[0].diff[10:15]
            print "fourth diff:"
            print top[0].diff[15:20]
            print "fifth diff:"
            print top[0].diff[20:25]
            return
        print "check diff:"
        print "shape:"
        print top[0].diff.shape
        print top[0].diff'''

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
