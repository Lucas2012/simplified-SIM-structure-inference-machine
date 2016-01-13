import unittest
import tempfile
import os
import numpy

import caffe

class Initial_Message(caffe.Layer):
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
    
    def reshape(self, bottom, top):
        # have one input one output, initialize messages for each node in the graphical model       
        bottom_batchsize = bottom[0].data.shape[0]
        slen = (self.nPeople+1)*self.nScene
        alen = self.nPeople*(self.message_num_action)*self.nAction
        tlen_leaf = 0
        tlen_mid = 0
        if (self.K_ > 0):
            tlen_leaf = 2*self.nPeople*2*self.nAction # 2 leaves, n people, each has 2 input messages
            self.tlen_leaf = tlen_leaf
        if (self.K_ > 1):
            tlen_mid = 2*(self.K_-1)*self.nPeople*3*self.nAction
            self.tlen_mid = tlen_mid
        top[0].reshape(bottom_batchsize, slen+alen+tlen_leaf+tlen_mid)
        self.bottom_batchsize = bottom_batchsize
        self.slen = slen
        self.alen = alen
        self.aunit = (self.message_num_action)*self.nAction
        self.tfunit = 2*self.nAction
        self.tmunit = 3*self.nAction

    def forward(self, bottom, top):
        # initialize messages for each node in the graphical model
        # put unary messages in and pad other messages to be one
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[1].data
        count = 0
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        sample_data = numpy.zeros([1,0])
        init_message1 = (1.0/self.nScene)*numpy.ones([1,(self.nPeople+1)*self.nScene])
        sample_data = numpy.append(init_message1,sample_data,axis = 1).copy()
        init_message2 = (1.0/self.nAction)*numpy.ones([1,self.nPeople*(self.message_num_action)*self.nAction])
        sample_data = numpy.append(sample_data,init_message2,axis = 1).copy()
        if (self.K_>0):
            init_message3_leaf = (1.0/self.nAction)*numpy.ones([1,2*self.nPeople*2*self.nAction])
            sample_data = numpy.append(sample_data,init_message3_leaf,axis = 1).copy()
        if (self.K_>1):
            init_message3.mid = (1.0/self.nAction)*numpy.ones([1,2*(self.K_-1)*self.nPeople*3*self.nAction])
            sample_data = numpy.append(sample_data,init_message3_mid,axis = 1).copy()
        #print sample_data[0:10]


        inits = (self.nPeople+1)*self.nScene
        #convenience = self.message_num_action*self.nAction
        tmpdataall = top[0].data
        for i in range(0,self.bottom_batchsize):
            tmpdataall[i] = numpy.zeros(sample_data.shape)
        for i in range(0,self.bottom_batchsize):
            length1 = sample_data.shape
            #mask = numpy.ones([1,length1[1]])  #cannot handle when K!=0, should set temporal mask to 0s
            true_action = label_stop[i]*self.nAction
            full_action = self.nPeople*self.nAction
            '''mask[0,inits+self.aunit*label_stop[i]:inits+self.aunit*self.nPeople] = numpy.zeros([1,(self.nPeople-label_stop[i])*self.aunit])
            mask[0,label_stop[i]*self.nScene:self.nPeople*self.nScene] = numpy.zeros([1,(self.nPeople-label_stop[i])*self.nScene])
            tmpdataall[i] = numpy.multiply(sample_data,mask)'''

            tmpdataall[i] = sample_data.copy()

            '''# avereage/zero initialization:
            a = bottom[0].data[i,0:self.nScene]
            tmpdataall[i,self.nPeople*self.nScene:self.slen] = a'''
            
            # repeated initialization:
            sceneunary = numpy.reshape(bottom[0].data[i,0:self.nScene],[1,self.nScene]).copy()
            fortmp = numpy.repeat(sceneunary,self.nPeople+1,axis = 0).copy()
            #print sceneunary
            #print fortmp
            fortmp = numpy.reshape(fortmp,[1,self.slen]).copy()
            tmpdataall[i,0:self.slen] = fortmp[0].copy()
            tmpdataall[i,self.nScene*label_stop[i]:self.slen-self.nScene] = numpy.zeros([1,(self.nPeople-label_stop[i])*self.nScene])
            #print i,i,i,i,i
            #print tmpdataall[1,0:self.slen]
            stepb = self.nScene
            step = self.slen
            for j in range(0,self.nPeople):
                if j >= label_stop[i]:
                    tmpdataall[i,step+j*self.aunit:step+j*self.aunit+self.aunit] = numpy.zeros([1,self.aunit])
                    continue
                j_aunit = j*self.aunit
                j_action = j*self.nAction


                '''# the average initialization:
                tmpdataall[i,step+j_aunit+true_action:step+j_aunit+full_action] = numpy.zeros([1,full_action - true_action])
                tmpdataall[i,step+j_aunit+j_action:step+j_aunit+j_action+self.nAction] = bottom[0].data[i,stepb+j_action:stepb+j_action+self.nAction]'''

                # the repeated initialization:
                #  ---> !! the repeated initialization not done for temporal version!!
                unary = bottom[0].data[i,stepb+j_action:stepb+j_action+self.nAction].copy()
                unary = numpy.reshape(unary,[1,self.nAction]).copy()
                fortmp = numpy.repeat(unary,self.message_num_action,axis = 0).copy()
                fortmp = numpy.reshape(fortmp,[1,self.aunit]).copy()
                tmpdataall[i,step+j_aunit:step+j_aunit+self.message_num_action*self.nAction] = fortmp[0].copy()
                tmpdataall[i,step+j_aunit+true_action:step+j_aunit+full_action] = numpy.zeros([1,full_action - true_action])                


                if (self.K_<1):
                    continue;
                leaflen = self.tlen_leaf/2
                step = step + self.alen
                stepb = stepb + self.nPeople*self.nAction
                for lf in range(0,2):
                    tmpdataall[i,step+j*self.tfunit:step+j*self.tfunit + self.nAction] = bottom[0].data[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction].copy()
                    step = step+leaflen
                    stepb = stepb+self.nPeople*self.nAction

                if (self.K_<2):
                    continue
                midlen = self.tlen_mid/(2*(self.K_-1))
                for mid in range(0,2*(self.K_-1)):
                    tmpdataall[i,step+j*self.tmunit:step+j*self.tmunit+self.nAction] = bottom[0].data[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction].copy()
                    step = step + midlen
                    stepb = stepb + self.nPeople*self.nAction
        top[0].data[...] = tmpdataall.copy()    
        #print tmpdataall[0,0:self.slen]
        #assert(1==0)            

    def backward(self, top, propagate_down, bottom):
        # the initialized messages have no bottom to be backpropogated, 
        # simply pick out the unaries to backprop
        tmp_diff = bottom[0].diff.copy()
        
        for i in range(0,self.bottom_batchsize):
            tmp_diff[i,0:self.nScene] = top[0].diff[i,self.nPeople*self.nScene:self.slen].copy()
            # temporarily remove action diff:
            # continue
            for j in range(0,self.nPeople):
                if j >= self.label_stop[i]:
                    break;
                step = self.slen
                stepb = self.nScene
                tmp_diff[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction] = top[0].diff[i,step+j*self.aunit+j*self.nAction:step+j*self.aunit+(j+1)*self.nAction].copy()
                
                if (self.K_<1):
                    continue
                leaflen = self.tlen_leaf/2
                step = step + self.alen
                stepb = stepb + self.nPeople*self.nAction
                for lf in range(0,2):
                    tmp.diff[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction] = top[0].diff[i,step+j*self.tfunit:step+j*self.tfunit + self.nAction].copy()
                    step = step + leaflen
                    stepb = stepb + self.nPeople*self.nAction

                if (self.K_<2):
                    continue
                midlen = self.tlen_mid/(2*(self.K_-1))
                for mid in range(0,2*(self.K_-1)):
                    bottom[0].diff[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction] = top[0].diff[i,step+j*self.tmunit:step+j*self.tmunit+self.nAction].copy()
                    step = step + midlen
                    stepb = stepb + self.nPeople*self.nAction

        bottom[0].diff[...] = tmp_diff.copy() 
        #print tmp_diff[0]

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
