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
        # 1 is compare unary input and pred_message; 2 is compare pred_message with other messages
        self.strategy = 1
        self.id = 0
        self.minus_s = False
    
    def reshape(self, bottom, top):
        # have one input one output, initialize messages for each node in the graphical model       
        bottom_batchsize = bottom[0].data.shape[0]
        edge_num = self.nPeople*(self.nPeople-1)/2
        self.bottom_batchsize = bottom_batchsize
        self.top_batchsize = bottom_batchsize*edge_num*2
        if self.minus_s:
            top[0].reshape(self.top_batchsize, self.nAction)
        else:
            top[0].reshape(self.top_batchsize, self.nAction*2)

    def forward(self, bottom, top):
        unary_input = bottom[0].data.copy()
        tms1_message = bottom[1].data.copy()
        edge_num = self.nPeople*(self.nPeople-1)/2
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[2].data
        count = 0
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        # the paired up inputs should be:
        # [(1,2),(2,1)]   [(1,3),(3,1)]   [(1,4),(4,1)]   [(1,5),(5,1)]   [(1,6),(6,1)]
        # [(2,3),(3,2)]   [(2,4),(4,2)]   [(2,5),(5,2)]   [(2,6),(6,2)]
        # [(3,4),(4,3)]   [(3,5),(5,3)]   [(3,6),(6,3)]
        # [(4,5),(5,4)]   [(4,6),(6,4)]
        # [(5,6),(6,5)]
        tmpdata = top[0].data
        output_c = 0
        count = 0
        self.id = self.id%3 + 1
        minus_s = self.minus_s
        for f in range(0,self.bottom_batchsize):
            for i in range(0,self.nPeople):
                m_scene = tms1_message[f,75+(i+1)*self.message_num_action*self.nAction-self.nAction:75+(i+1)*self.message_num_action*self.nAction]
                if i < self.label_stop[f] and output_c < 2:
                    #print m_scene
                    output_c += 1
                u_action_pi = unary_input[f,5+i*self.nAction:5+(i+1)*self.nAction]
                for j in range(i+1,self.nPeople):
                    u_action_pj =  unary_input[f,5+j*self.nAction:5+(j+1)*self.nAction]
                    if i >= self.label_stop[f] or j >= self.label_stop[f]:
                        if minus_s:
                            blank = numpy.zeros(self.nAction) 
                        else:
                            blank = numpy.zeros(2*self.nAction) 
                        tmpdata[count] = blank
                        count += 1
                        tmpdata[count] = blank
                        count += 1
                        continue
                    start_j = 75+j*self.message_num_action*self.nAction + i*self.nAction
                    end_j = 75+j*self.message_num_action*self.nAction + (i+1)*self.nAction
                    start_i = 75+i*self.message_num_action*self.nAction + j*self.nAction
                    end_i = 75+i*self.message_num_action*self.nAction + (j+1)*self.nAction
                    # predicted message:
                    m_action_ji = tms1_message[f,start_j:end_j]  # messages around node j passed from node i 
                    m_action_ij = tms1_message[f,start_i:end_i]  # messages around node i passed from node j
                    predicted_m_ji = m_action_ij
                    predicted_m_ij = m_action_ji
                    if self.strategy == 1:
                        reference_m_i = u_action_pi
                        reference_m_j = u_action_pj
                    else:
                        sumj = numpy.sum(numpy.reshape(tms1_message[f,75+j*self.message_num_action*self.nAction:75+(j+1)*self.message_num_action*self.nAction],[15,self.nAction]))
                        sumi = numpy.sum(numpy.reshape(tms1_message[f,75+i*self.message_num_action*self.nAction:75+(i+1)*self.message_num_action*self.nAction],[15,self.nAction]))
                        sumj -= m_action_ji
                        sumj /= self.label_stop[f]
                        sumi -= m_action_ij
                        sumi /= self.label_stop[f]
                        reference_m_i = sumi
                        reference_m_j = sumj
                    # for minus: figure out if use absolute value or not!
                    if minus_s == False:
                        #pairij = numpy.append(reference_m_i,predicted_m_ij)
                        #pairji = numpy.append(reference_m_j,predicted_m_ji)
                        pairij = numpy.append(predicted_m_ij,predicted_m_ji)
                        pairji = numpy.append(predicted_m_ij,predicted_m_ji)
                    else:
                        pairij = reference_m_i - predicted_m_ij
                        pairji = reference_m_j - predicted_m_ji
                             
                    tmpdata[count] = pairij
                    count += 1
                    tmpdata[count] = pairji
                    count += 1
        top[0].data[...] = tmpdata       

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
                        count += 2
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
