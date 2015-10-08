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
        self.T_ = 1
        self.label_stop = []
        self.count = 0
    
    def reshape(self, bottom, top):
        # have one input one output, initialize messages for each node in the graphical model
        bottom_shape = bottom[0].data.shape
        self.bottom_batchsize = bottom_shape[0]/self.T_  # num of frames
        self.bottom_output_num = bottom_shape[1]  # naction
        labels=bottom[1].data
        #print labels
        label_stop = numpy.ones([self.bottom_batchsize])
        count = 0
        #print bottom_shape
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    count += j
                    break
        self.count = count
        if self.count == 0:
            self.count = self.bottom_batchsize
        self.label_stop = label_stop
        self.top_batchsize = self.count*self.T_
        self.top_output_num = self.bottom_output_num/self.nPeople
        top[0].reshape(self.top_batchsize, self.top_output_num)
        top[1].reshape(self.top_batchsize)
        if len(bottom) > 2:
            top[2].reshape(self.top_batchsize)

    def forward(self, bottom, top):
        #top[0].data = numpy.reshape(bottom[0].data,[self.top_batchsize,self.bottom_output_num])
        #top[1].data = numpy.repeat(bottom[1].data,self.T_)
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[1].data
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        if len(bottom) > 2:
            labels2 = bottom[2].data
        else:
            labels2 = bottom[1].data
        labels = numpy.reshape(labels,[self.bottom_batchsize,self.nPeople])
        labels2 = numpy.reshape(labels2,[self.bottom_batchsize,self.nPeople])
        tmpdata = numpy.zeros([0,self.nAction])
        tmplabel = numpy.zeros([0,0])
        tmplabel2 = numpy.zeros([0,0])
        for i in range(0,self.T_):
            for j in range(0,self.bottom_batchsize):
                #print bottom[0].data.shape
                tmp = numpy.reshape(bottom[0].data[i*self.bottom_batchsize+j],[self.nPeople,self.nAction]).copy()
                tmpdata = numpy.append(tmpdata,tmp[0:self.label_stop[j]],axis = 0).copy()
                tmplabel = numpy.append(tmplabel,labels[j,0:self.label_stop[j]]).copy()
                tmplabel2 = numpy.append(tmplabel2,labels2[j,0:self.label_stop[j]]).copy()
                #print tmpdata
                #print tmplabel
        #tmplabel=numpy.reshape(tmplabel,[len(tmplabel),1])
        #tmplabel2=numpy.reshape(tmplabel2,[len(tmplabel2),1])
        top[0].data[...] = tmpdata
        top[1].data[...] = tmplabel+1.0
        if len(bottom) > 2:
            top[2].data[...] = tmplabel2
            #print tmplabel2
        #print tmplabel
        #print tmpdata
        '''print "orin_label"
        print bottom[1].data
        print "orin_label"
        print "label"
        print top[1].data
        print "label"'''

    def backward(self, top, propagate_down, bottom):
        topdiff = top[0].diff
        topdiff = numpy.reshape(topdiff,[self.T_,self.count,self.nAction])
        tmpdiff = numpy.zeros(bottom[0].data.shape)
        for i in range(0,self.T_):
            step = 0
            for j in range(0,self.bottom_batchsize):
                assert(self.label_stop[j] > 0)
                tmpuse = topdiff[i,step:step+self.label_stop[j]].copy()
                tmpuse = numpy.reshape(tmpuse,[1,self.label_stop[j]*self.nAction]).copy()
                #print tmpuse
                #print tmpuse[0]
                #assert(tmpuse[0,0]!=0)
                tmpdiff[i*self.bottom_batchsize+j,0:self.label_stop[j]*self.nAction] = tmpuse.copy()
                step += self.label_stop[j]
                #print "stop:",self.label_stop[j]
                #print tmpdiff[i*self.bottom_batchsize+j]
        #print "topshape:",top[0].diff.shape
        #print "step:",step
        bottom[0].diff[...] = tmpdiff.copy()
        tmpdiff = numpy.reshape(tmpdiff,[bottom[0].diff.shape[0]*self.nPeople,self.nAction]).copy()
        '''print "check original"
        print "1"
        print bottom[0].diff[0:5]
        print "2"
        print bottom[0].diff[5:10]
        print "3"
        print bottom[0].diff[10:15]
        print "4"
        print bottom[0].diff[15:20]
        print "5"
        print bottom[0].diff[20:25]'''
        #print tmpdiff[:,0:1]
        #assert(bottom[0].diff)
        #print tmpdiff

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
