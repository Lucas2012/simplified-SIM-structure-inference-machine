import unittest
import tempfile
import os
import numpy

import caffe

class Data_Arrange_Layer(caffe.Layer):
    """A layer that initialize messages for recurrent belief propagation"""

    def setup(self, bottom, top):
        self.nScene = 5
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
        self.frame_ = 1
        self.label_stop = []
        self.count = 0
    
    def reshape(self, bottom, top):
        # have one input one output, initialize messages for each node in the graphical model
        bottom_shape = bottom[0].data.shape
        self.frame_ = bottom_shape[0]/self.nPeople  # num of frames
        self.bottom_output_num = bottom_shape[1]  # naction
        labels=bottom[1].data
        
        label_stop = numpy.ones([self.frame_])
        count = 0

        for i in range(0,self.frame_):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    count += j
                    break
        self.count = count
        if self.count == 0:
            self.count = self.frame_
        #print self.frame_
        self.label_stop = label_stop
        self.top_batchsize = self.count
        self.top_output_num = self.bottom_output_num
        top[0].reshape(self.top_batchsize, self.top_output_num)
        top[1].reshape(self.top_batchsize)
        #top[2].reshape(self.top_batchsize)

    def forward(self, bottom, top):
        #top[0].data = numpy.reshape(bottom[0].data,[self.top_batchsize,self.bottom_output_num])
        #top[1].data = numpy.repeat(bottom[1].data,self.T_)
        labels = bottom[1].data
        #labels2 = bottom[2].data
        labels = numpy.reshape(labels,[self.frame_,self.nPeople])
        #labels2 = numpy.reshape(labels2,[self.bottom_batchsize,self.nPeople])
        tmpdata = numpy.zeros([0,self.nAction])
        tmplabel = numpy.zeros([0,0])
        #tmplabel2 = numpy.zeros([0,0])
        #print self.frame_
        for j in range(0,self.frame_):
            tmp = bottom[0].data[j*self.nPeople:(j+1)*self.nPeople]
            #print tmp
            tmpdata = numpy.append(tmpdata,tmp[0:self.label_stop[j]],axis = 0)
            tmplabel = numpy.append(tmplabel,labels[j,0:self.label_stop[j]])
            #tmplabel2 = numpy.append(tmplabel2,labels2[j,0:self.label_stop[j]])
        #tmplabel=numpy.reshape(tmplabel,[len(tmplabel),1])
        #tmplabel2=numpy.reshape(tmplabel2,[len(tmplabel2),1])
        top[0].data[...] = tmpdata
        top[1].data[...] = tmplabel
        #top[2].data[...] = tmplabel2
        #print tmplabel
        #print tmpdata

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
