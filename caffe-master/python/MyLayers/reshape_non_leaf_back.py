import unittest
import tempfile
import os

import caffe

class reshape_non_leaf_back(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.nAction = 6
        self.nPeople = 10
        #self.weight = [0] * self.nAction * self.nPeople
        # shared weights
        #could initialize weights using gaussian function
        self.flag = 0
        self.data = []
        self.fakedata = [0] * self.nPeople*(self.nAction+1)
        # read the file containing number of people in each frame
        self.frame_people = [2,3,4,5,6,7,8,1,2,6,2,3,4,5,6,7,8,1,2,6,2,3,4,5,6,7,8,1,2,6,2,3,4,5,6,7,8,1,2,6,]
        self.count = 0
        #f = fopen("number_of_people.txt")
        # initialize frame_people
    
    def reshape(self, bottom, top):
        print bottom[0].data.shape
        top[0].reshape(*bottom[0].data.shape)

    def forward(self, bottom, top):
        print bottom[0].data
        if self.count < self.frame_people[self.flag]:
            self.data.append(bottom[0].data)
            top[0].data = self.fakedata
            print len(self.data)
        else:
            self.count = 0
            top[0].data = self.data
            self.data = []
            self.flag = self.flag+1
        '''if sum(bottom[0].data) != 0:
            self.data.append(bottom[0].data)
            top[0].data = self.fakedata
        else:
            top[0].data = self.data
            self.data = []'''

    def backward(self, top, propagate_down, bottom):
        print propagate_down[0]
        tmp_bottom = zeros([1,len(bottom[0].diff)])
        nPeople = top[0].data/(1+self.nAction)
        for i in range(0,len(top[0].diff)):
            ac = i % (nAction+1)
            tmp_bottom[ac] += top[0].diff[i]
        bottom[0].diff = tmp_bottom

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
