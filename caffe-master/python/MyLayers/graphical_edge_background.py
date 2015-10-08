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
        # 1 is compare unary input and pred_message; 2 is compare two unaries; 3 is compare pred_message with other messages
        self.strategy = 1
        self.id = 0
        self.minus_s = False
        self.all_message = True
    
    def reshape(self, bottom, top):
        # have 4 inputs: bottom0 is unary inputs, bottom1 is a2s predictions, bottom2 is s2a predictions, bottom3 is labels    
        # have 1 outputs: top0 for background gates input 
        bottom_batchsize = bottom[0].data.shape[0]
        edge_num = self.nPeople
        self.bottom_batchsize = bottom_batchsize
        self.top_batchsize = bottom_batchsize*edge_num
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[3].data
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0 and j != 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        self.top_batchsize = int(numpy.sum(label_stop))
        if self.minus_s or self.strategy == 2:
            action_information_len = 0#self.nScene#self.nAction
            scene_information_len = self.nScene
        else:
            action_information_len = self.nAction*2
            scene_information_len = self.nScene*2       
        top[0].reshape(self.top_batchsize, action_information_len + scene_information_len)

    def forward(self, bottom, top):
        unary_input = bottom[0].data.copy()
        a2s_pred = bottom[1].data.copy()
        s2a_pred = bottom[2].data.copy()
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[3].data
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
        minus_s = self.minus_s
        step = 0
        for f in range(0,self.bottom_batchsize):
            a2s_pred_f = a2s_pred[step:step+label_stop[f]]
            s2a_pred_f = s2a_pred[step:step+label_stop[f]]
            scene_pred_all = a2s_pred_f.sum(axis = 0)
            action_pred_all = s2a_pred_f.sum(axis = 0)
            step += label_stop[f]
            for p in range(0,self.nPeople):
                if self.strategy == 1:
                    if p >= self.label_stop[f]:
                        continue
                    scene_unary = unary_input[f,:self.nScene].copy()
                    scene_pred = a2s_pred[p].copy()
                    action_unary = unary_input[f,self.nScene+p*self.nAction:self.nScene+(p+1)*self.nAction]
                    action_pred = s2a_pred[p].copy()
                    if self.all_message:
                        #scene_unary += scene_pred_all
                        scene_unary = scene_pred_all
                        scene_unary -= scene_pred
                        scene_unary /= (1.0*label_stop[f])
                        #action_unary += action_pred_all
                        action_unary = action_pred_all
                        action_unary -= action_pred
                        action_unary /=(1.0*label_stop[f])
                    scene_potential = numpy.append(scene_unary,scene_pred,axis = 0).copy()
                    action_potential = numpy.append(action_unary,action_pred,axis = 0).copy()
                    edge_potential = numpy.append(scene_potential,action_potential,axis = 0).copy()
                    tmpdata[count] = edge_potential.copy()
                    count += 1
                    #print 'action_pred',action_pred
                    #print 'action_unary',action_unary
                    continue
                if self.strategy == 2:
                    if p >= self.label_stop[f]:
                        continue
                    scene_unary = unary_input[f,:self.nScene].copy()
                    action_unary = unary_input[f,self.nScene+p*self.nAction:self.nScene+(p+1)*self.nAction].copy()
                    #edge_potential = numpy.append(scene_unary,action_unary[2:],axis = 0)
                    edge_potential = scene_unary-action_unary[2:]
                    tmpdata[count] = edge_potential.copy()
                    #print 'scene',scene_unary
                    #print 'action',action_unary
                    #print tmpdata[count]
                    count += 1
                    continue
        top[0].data[...] = tmpdata    
          

    def backward(self, top, propagate_down, bottom):
        return
       
             
        

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
