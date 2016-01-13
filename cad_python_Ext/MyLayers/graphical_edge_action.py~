import unittest
import tempfile
import os
import numpy

import caffe

class graphical_edge(caffe.Layer):
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
        self.message_num_action = self.nPeople+1+2*(self.K_>0)
        self.label_stop = []
        self.top_batchsize = 0
        # 1 is compare unary input and pred_message; 2 is compare pred_message with other messages; 3 is compare pred_message with other messages and unary
        self.strategy = 3
        self.id = 0
        self.minus_s = False
    
    def reshape(self, bottom, top):
        # have 3(4) inputs: unary actions, a2a messages, labels, and (optional) distance between people      
        bottom_batchsize = bottom[0].data.shape[0]
        edge_num = self.nPeople
        self.bottom_batchsize = bottom_batchsize
        self.top_batchsize = bottom[1].data.shape[0]
        if self.minus_s:
            if len(bottom) > 3 and bottom[3].data.shape[1] == self.nPeople:
                top[0].reshape(self.top_batchsize, self.nAction+1)
            else:
                top[0].reshape(self.top_batchsize, self.nAction)
        else:
            if len(bottom) > 3  and bottom[3].data.shape[1] == self.nPeople:
                if self.strategy == 3:
                    top[0].reshape(self.top_batchsize, self.nAction*3+1)
                elif self.strategy == 1 or self.strategy == 2:
                    top[0].reshape(self.top_batchsize, self.nAction*2+1)
            else:
                if self.strategy == 3:
                    top[0].reshape(self.top_batchsize, self.nAction*3)
                elif self.strategy == 1 or self.strategy == 2:
                    top[0].reshape(self.top_batchsize, self.nAction*2)

    def forward(self, bottom, top):
        unary_input = bottom[0].data.copy()
        a2a_message_pred = bottom[1].data.copy()
        #print a2a_message_pred.shape
        #print 'a2apred',a2a_message_pred
        
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[2].data

        #print 'labels',labels
        #for i in range(0,a2a_message_pred.shape[0]):
        #    print 'action in',bottom[3].data[i]
        #    print 'a2a_pred',a2a_message_pred[i]

        if len(bottom) > 3:
            distance = bottom[3].data.copy()
        count = 0
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        tmpdata = top[0].data
        count = 0
        step = 0
        minus_s = self.minus_s
        #print self.bottom_batchsize
        #print label_stop[0]
        for f in range(0,self.bottom_batchsize):
            for i in range(0,self.nPeople):
                if i >= label_stop[f] or label_stop[f] == 1:
                    continue
                f_a2a_message_pred = a2a_message_pred[step:step+self.label_stop[f]-1]
                a2a_pred_all = f_a2a_message_pred.sum(axis = 0)
                for j in range(0,self.nPeople):
                    if i == j or j >=label_stop[f]:
                        continue
                    unary = unary_input[f,self.nScene+i*self.nAction:self.nScene+(i+1)*self.nAction].copy()

                    pair = numpy.append(unary,a2a_message_pred[count],axis = 0).copy()
                    if self.strategy == 3:
                        if self.label_stop[f]-2 == 0:
                            pair = numpy.append(pair,a2a_message_pred[count],axis = 0).copy()
                        else:
                            others = (a2a_pred_all-a2a_message_pred[count])/(self.label_stop[f]-2)
                            pair = numpy.append(pair,others,axis = 0).copy()
                    if len(bottom) > 3 and bottom[3].data.shape[1] == self.nPeople:
                        pair = numpy.append(pair,distance[i+f*self.nPeople,j])
                    tmpdata[count] = pair.copy()
                    count += 1
                step += self.label_stop[f]-1
        top[0].data[...] = tmpdata 
        #print tmpdata
             

    def backward(self, top, propagate_down, bottom):
        return
        # strategy1: unary input and pred message
        # strategy2: pred message and average of other messages
        # minus_s: if take the two input vectors as a concatenation or subtraction
        unary_diff = bottom[0].diff
        tms1_message_diff = bottom[1].diff
        top_diff = top[0].diff
        
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[2].data
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        minus_s = self.minus_s
        count = 0
        for f in range(0,self.bottom_batchsize):
            for i in range(0,self.nPeople):
                for j in range(i+1,self.nPeople):
                    if i >= self.label_stop[f] or j >= self.label_stop[f]:
                        count += 1
                        continue
                    diffij = top_diff[count]
                    diffji = top_diff[count+1]
                    count += 2
                    if minus_s:
                        reference_m_i_diff = diffij
                        reference_m_j_diff = diffji
                        pred_m_ij_diff = -diffij
                        pred_m_ji_diff = -diffji
                    else:
                        reference_m_i_diff = diffij[0:self.nAction]
                        reference_m_j_diff = diffji[0:self.nAction]
                        pred_m_ij_diff = diffij[self.nAction:]
                        pred_m_ji_diff = diffji[self.nAction:]
                    if self.strategy == 1:
                        bottom[0].diff[f,self.nScene+self.nAction*i:self.nScene+self.nAction*(i+1)] += reference_m_i_diff
                        bottom[0].diff[f,self.nScene+self.nAction*j:self.nScene+self.nAction*(j+1)] += reference_m_j_diff
                    else:
                        reference_i = numpy.repeat([reference_m_i_diff],self.message_num_action,axis = 0)/self.label_stop[f]
                        reference_j = numpy.repeat([reference_m_j_diff],self.message_num_action,axis = 0)/self.label_stop[f]
                        reference_i[j] = numpy.zeros(self.nAction)
                        reference_j[i] = numpy.zeros(self.nAction)
                        reference_i[self.label_stop[f]:self.nPeople] = numpy.zeros([self.nPeople-self.label_stop[f],self.nAction])
                        reference_j[self.label_stop[f]:self.nPeople] = numpy.zeros([self.nPeople-self.label_stop[f],self.nAction]) 
                        start_j = 75+j*self.message_num_action*self.nAction
                        end_j = 75+(j+1)*self.message_num_action*self.nAction
                        start_i = 75+i*self.message_num_action*self.nAction
                        end_i = 75+(i+1)*self.message_num_action*self.nAction
                        bottom[1].diff[f,start_i:end_i] += numpy.reshape(reference_i,[1,len(reference_m_i_diff)*len(reference_i)])[0]
                        bottom[1].diff[f,start_j:end_j] += numpy.reshape(reference_j,[1,len(reference_m_j_diff)*len(reference_j)])[0]
                    bottom[1].diff[f,start_j+i*self.nAction:start_j+(i+1)*self.nAction] += pred_m_ji_diff
                    bottom[1].diff[f,start_i+j*self.nAction:start_i+(j+1)*self.nAction] += pred_m_ij_diff
                    
             
        

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
