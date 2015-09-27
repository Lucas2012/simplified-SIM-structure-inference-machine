import unittest
import tempfile
import os
import numpy

import caffe

class test_Initial_Message(caffe.Layer):
    """A layer that initialize messages for recurrent belief propagation"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 40
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
        sample_data = numpy.append(init_message1,sample_data,axis = 1)
        init_message2 = (1.0/self.nAction)*numpy.ones([1,self.nPeople*(self.message_num_action)*self.nAction])
        sample_data = numpy.append(sample_data,init_message2,axis = 1)
        if (self.K_>0):
            init_message3_leaf = (1.0/self.nAction)*numpy.ones([1,2*self.nPeople*2*self.nAction])
            sample_data = numpy.append(sample_data,init_message3_leaf,axis = 1)
        if (self.K_>1):
            init_message3.mid = (1.0/self.nAction)*numpy.ones([1,2*(self.K_-1)*self.nPeople*3*self.nAction])
            sample_data = numpy.append(sample_data,init_message3_mid,axis = 1)
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

            tmpdataall[i] = sample_data

            '''# avereage/zero initialization:
            a = bottom[0].data[i,0:self.nScene]
            tmpdataall[i,self.nPeople*self.nScene:self.slen] = a'''
            
            # repeated initialization:
            sceneunary = numpy.reshape(bottom[0].data[i,0:self.nScene],[1,self.nScene])
            fortmp = numpy.repeat(sceneunary,self.nPeople+1,axis = 0)
            fortmp = numpy.reshape(fortmp,[1,self.slen])
            tmpdataall[i,0:self.slen] = fortmp[0]
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
                unary = bottom[0].data[i,stepb+j_action:stepb+j_action+self.nAction]
                unary = numpy.reshape(unary,[1,self.nAction])
                fortmp = numpy.repeat(unary,self.message_num_action,axis = 0)
                fortmp = numpy.reshape(fortmp,[1,self.aunit])
                tmpdataall[i,step+j_aunit:step+j_aunit+self.message_num_action*self.nAction] = fortmp[0]
                tmpdataall[i,step+j_aunit+true_action:step+j_aunit+full_action] = numpy.zeros([1,full_action - true_action])                


                if (self.K_<1):
                    continue;
                leaflen = self.tlen_leaf/2
                step = step + self.alen
                stepb = stepb + self.nPeople*self.nAction
                for lf in range(0,2):
                    tmpdataall[i,step+j*self.tfunit:step+j*self.tfunit + self.nAction] = bottom[0].data[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction]
                    step = step+leaflen
                    stepb = stepb+self.nPeople*self.nAction

                if (self.K_<2):
                    continue
                midlen = self.tlen_mid/(2*(self.K_-1))
                for mid in range(0,2*(self.K_-1)):
                    tmpdataall[i,step+j*self.tmunit:step+j*self.tmunit+self.nAction] = bottom[0].data[i,stepb+j*self.nAction:stepb+(j+1)*self.nAction]
                    step = step + midlen
                    stepb = stepb + self.nPeople*self.nAction
        top[0].data[...] = tmpdataall  
        # print   "initial:",top[0].data
        #print tmpdataall[0,0:self.slen]
        #assert(1==0)            

    def backward(self, top, propagate_down, bottom):
        # the initialized messages have no bottom to be backpropogated, 
        # simply pick out the unaries to backprop
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[1].data
        #print labels
        count = 0
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop
        tmp_diff = bottom[0].diff.copy()
        # print "diff:", tmp_diff
        #print "start"
        for i in range(0,self.bottom_batchsize):
            #print "i:",i
            tmp_diff[i,0:self.nScene] = top[0].diff[i,self.nPeople*self.nScene:self.slen].copy()
            # temporarily remove action diff:
            # continue
            #print "cp1"
            for j in range(0,self.nPeople):
                #print "j:",j
                #print "labelstop:",self.label_stop[i]
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
        input: 'data' input_shape { dim: 5 dim: 565 }
        input: 'label' input_shape {dim: 70}
        layer { type: 'Python' name: 'one' bottom: 'data' bottom: 'label' top: 'one'
          python_param { module: 'test_Initialize_Message' layer: 'test_Initial_Message' } }""")
        return f.name

class TestInitializeMessage(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        #input shape: scene classes 5 + num of people 14 * action classes 40
        lengthx = 5 + 40*14
        x_scene = [-5,-4,-3,-2,-1]
        x_action = list(xrange(1,40*14+1))
        x_unit = x_scene+x_action
        l = [];
        x = [];
        for a in range(0,5):
            x.append(x_unit)
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            l = l + tmp
        self.net.blobs['data'].data[...] = x
        self.net.blobs['label'].data[...] = l
        self.net.forward()

        # design number of people in each frame:
        labelst = [1,2,3,4,14]

        # generate ground truth for scene scores:
        scene_all = []
        for batch in range(0,5):
            tmp_scene = labelst[batch]*x_scene + [0]*5*(14-labelst[batch]) + x_scene
            scene_all.append(tmp_scene)

        # generage ground truth for all actions in a scene
        action_all = []
        for batch in range(0,5):
            action_frame = x[batch]
            tmp_action_all = []
            for people in range(0,14):
                if people < labelst[batch]:
		            action_per = action_frame[40*people+5:40*(people+1)+5]
		            # replicate corresponding number of messages around each node
		            tmp_action = action_per*labelst[batch] + [0]*40*(14-labelst[batch]) + action_per
                else:
                    tmp_action = [0]*40*15
                tmp_action_all += tmp_action
            action_all.append(tmp_action_all)
            
        # concatenate initialized
        gt_position = numpy.append(scene_all,action_all,axis = 1)

        # assert equal for this layer's output
        for batch1,batch2 in zip(self.net.blobs['one'].data,gt_position):
            # for every batch
            stop_id = labelst[batch]
            for x,y in zip(batch1,batch2):
                self.assertEqual(x, y)

    def test_backward(self):
        # label stop:
        labelst = [1,2,3,4,14]
        l = [];
        for a in range(0,5):
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            l = l + tmp
        self.net.blobs['label'].data[...] = l
        # top diffs:  length 7910 = 5*(14+1) + 14*(14+1)*40 = 75 + 8400
        x_unit = list(xrange(0,8475))
        x = []
        for i in range(0,5):
            x.append(x_unit)

        # generate ground truth scene diffs:
        scene_diff = []
        for i in range(0,5):
            scene_x = x[i]
            scene_diff.append(scene_x[70:75])        

        # generate ground truth action diffs:
        action_diff = []
        for i in range(0,5):
            per_frame = []
            for j in range(0,14):
                if j < labelst[i]:
                    action_x = x[i]
                    action_unary = action_x[j*40+75+j*600:(j+1)*40+75+j*600]
                else:
                    action_unary = [0]*40    
                per_frame += action_unary
            action_diff.append(per_frame)

        # concatenate scene and action
        gt_position = numpy.append(scene_diff,action_diff,axis = 1)

        # the bottom diff should equal to specific positions
        self.net.blobs['one'].diff[...] = x
        self.net.backward()
        pos = 0
        for batch1,batch2 in zip(self.net.blobs['data'].diff,gt_position):
            #print pos
            pos+=1
            #print batch2
            for x,y in zip(batch1,batch2):
                self.assertEqual(y, x)

    def test_reshape(self):
        # skip this test
        return
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
