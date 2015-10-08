import unittest
import tempfile
import os
import numpy

import caffe

class structured_gate(caffe.Layer):
    """A layer that initialize messages for recurrent belief propagation"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0;
        self.bottom_batchsize = 0
        self.slen = 0
        self.alen = 0
        self.tlen_leaf = 0
        self.tlen_mid = 0
        self.sunit = 0
        self.aunit = 0
        self.tunit = 0
        self.label_stop = []
        self.top_batchsize = 0
        self.on_edge = True
        self.zero2one = True
        self.frame_num = 0
        self.lamda = 0.01
    
    def reshape(self, bottom, top):
        # have 4 input, bottom0 is gate input, bottom1 is a2s messages, bottom2 is s2a messages, bottom3 is labels 
        # have 2 outputs, top0 is gated a2s messages, top1 is gated s2a messages    
        bottom_batchsize = bottom[0].data.shape[0]
        edge_num = self.nPeople
        self.bottom_batchsize = bottom_batchsize
        self.frame_num = bottom[3].data.shape[0]/self.nPeople
        top[0].reshape(*bottom[1].data.shape)
        top[1].reshape(*bottom[2].data.shape)

    def forward(self, bottom, top):
        gate_input = bottom[0].data.copy()
        edge_num = self.nPeople
        a2s_messages = bottom[1].data.copy()
        s2a_messages = bottom[2].data.copy()
        label_stop = self.nPeople*numpy.ones([self.frame_num])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.frame_num):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        # gate design:
        on_edge = self.on_edge
        if on_edge:
            s_gate = numpy.zeros(self.bottom_batchsize)
        else:
            s_gate = numpy.zeros([self.bottom_batchsize,self.nAction])
        zero2one = self.zero2one
        idx = 0
        for f in range(0,self.bottom_batchsize):
            o = (1+numpy.tanh(gate_input[idx]))/2.0
            s_gate[idx] = o
            messages_a2s = a2s_messages[idx].copy()
            messages_s2a = s2a_messages[idx].copy()
            top[0].data[idx] = numpy.multiply(o,messages_a2s)
            top[1].data[idx] = numpy.multiply(o,messages_s2a)
            idx += 1

    def backward(self, top, propagate_down, bottom):
        # to be written
        # diffs for : bottom[0] -> paired gates input; bottom[1] -> messages
        # diffs from top: gated_messages
        label_stop = self.nPeople*numpy.ones([self.frame_num])
        labels = bottom[3].data
        count = 0
        for i in range(0,self.frame_num):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        gate_input = bottom[0].data.copy()
        gate_diff = bottom[0].diff.copy()
        a2s_messages = bottom[1].data
        a2s_message_diff = bottom[1].diff.copy()
        gated_a2s_message_diff = top[0].diff.copy()
        s2a_messages = bottom[2].data
        s2a_message_diff = bottom[2].diff.copy()
        gated_s2a_message_diff = top[1].diff.copy()

        # gate design:
        on_edge = self.on_edge
        
        zero2one = self.zero2one
        count = 0
        idx = 0
        for f in range(0,self.bottom_batchsize):
            o = (1+numpy.tanh(gate_input[idx]))/2.0
            message_a2s = a2s_messages[idx].copy()
            message_a2s_diff = a2s_message_diff[idx].copy()
            message_s2a = s2a_messages[idx].copy()
            message_s2a_diff = s2a_message_diff[idx].copy()
            # gate diff
            tanh_sq = numpy.multiply(o,o)
            g_d_a2s = numpy.multiply(message_a2s,gated_a2s_message_diff)
            g_d_s2a = numpy.multiply(message_s2a,gated_s2a_message_diff)
            gate_diff[idx] = (g_d_a2s.sum() + g_d_s2a.sum())*(1-tanh_sq)
            # a2s message diff
            a2s_mesasge_diff = numpy.multiply(o,gated_a2s_message_diff)
            # s2a message diff
            s2a_mesasge_diff = numpy.multiply(o,gated_s2a_message_diff)
            idx += 1
            #print message_s2a_diff
            #print o
        gate_input = bottom[0].data.copy()
        gate_diff += self.lamda*gate_input
        bottom[0].diff[...] = gate_diff
        bottom[1].diff[...] = a2s_message_diff
        bottom[2].diff[...] = s2a_message_diff
        
        

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
