import unittest
import tempfile
import os
import numpy
import caffe
#import h5py
class Message_In(caffe.Layer):
    """A layer that take Messages in and output Q (Q=5) type of messages for prediction"""

    def setup(self, bottom, top):
        # self.SceneM = []                     # N*(P+1)*5   
        # self.ActionM = []                   # N*P*(P+2+1)*40
        # self.TemporalM = []                 # N*[2*P*2+(K_-1)*2*P*3]
        self.nScene = 5
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0	;
        self.bottom_batchsize = 0
        self.slen = 0
        self.alen = 0
        self.tlen_leaf = 0
        self.tlen_mid = 0 
        self.aunit = 0
        self.top_shape1 = [] # scene->action
        self.top_shape2 = [] # action->scene
        self.top_shape3 = [] # action->action
        self.top_shape4 = [] # temporal_after->temporal_before
        self.top_shape5 = [] # temporal_before->temporal_after
        self.message_num_action = self.nPeople+2*(self.K_>0)+1
        self.label_stop = []
        self.ifnormalize = True
        self.ifprevious = False
        self.ifensemble = True
        self.id = 0
        self.prevhardgate = True
    
    def reshape(self, bottom, top):
        # should have 1 bottom message in and 5 top messages out, two scene messages top, one action message top, two temporal messages
        slen = (self.nPeople+1)*self.nScene
        alen = self.nPeople*self.message_num_action*self.nAction
        if (self.K_>0):
            tlen_leaf = 2*self.nPeople*2*self.nAction # 2 leaves, n people, each has 2 input messages
            self.tlen_leaf = tlen_leaf
        if (self.K_>1):
            tlen_mid = 2*(self.K_-1)*self.nPeople*3*self.nAction
            self.tlen_mid = tlen_mid
        bottom_batchsize = bottom[0].data.shape[0]
        
        top_shape1 = [bottom_batchsize*self.nPeople, self.nScene]
        top_shape2 = [bottom_batchsize*self.nPeople, self.nAction]
        top_shape3 = [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        if self.K_>0:
            top_shape4 = [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
            top[3].reshape(*top_shape4)
            self.top_shape4 = top_shape4
            top_shape5 = [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
            top[4].reshape(*top_shape5)
            self.top_shape5 = top_shape5
        top[0].reshape(*top_shape1)
        top[1].reshape(*top_shape2)
        top[2].reshape(*top_shape3)
        
        self.bottom_batchsize = bottom_batchsize
        self.slen = slen
        self.alen = alen
        self.top_shape1 = top_shape1
        self.top_shape2 = top_shape2
        self.top_shape3 = top_shape3
        self.aunit = self.message_num_action*self.nAction
        self.ifprevious = False

    def forward(self, bottom, top):
        # arrange message input toward scene node, comes from ActionM
        # arrange message input action nodes in current frame, comes from ActionM, TemporalM and SceneM
        # arrange message input along time chains, both forward and backward
        # ifensemble: weighted actions for action message pred and for scene message pred are from the last two blobs
        # ifensemble: bottom[-1] is for action to action, bottom[-2] is for action2scene and scene2action
        if len(bottom) <= 5:
            self.ifensemble = False;
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[1].data
        self.ifprevious = False
        if len(bottom) > 2 and self.prevhardgate:
            self.ifprevious = True
            scene_score_tms1 = bottom[2].data
            if len(bottom[3].data.shape) == 3:
                action_score_tms1 = numpy.reshape(bottom[3].data,[self.bottom_batchsize,self.nPeople*self.nAction]).copy()
                action_score_tms1_2 = numpy.reshape(bottom[4].data,[self.bottom_batchsize,self.nPeople*self.nAction]).copy()
            else:
                action_score_tms1 = bottom[3].data
                action_score_tms1_2 = bottom[4].data

        count = 0
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        if self.ifensemble:
            bottom_ms1_data = bottom[-1].data.copy()
            bottom_ms2_data = bottom[-2].data.copy()
            if bottom_ms1_data.shape[1] == self.nAction:
                bottom_ms1_data = numpy.reshape(bottom_ms1_data,[self.bottom_batchsize,self.nPeople*self.nAction])
        count1 = 0;
        count2 = 0;
        count3 = 0;
        count4 = 0;
        count5 = 0;
        s_a_message = numpy.zeros(self.top_shape1)
        a_s_message = numpy.zeros(self.top_shape2)
        a_a_message = numpy.zeros(self.top_shape3)
        if self.K_ > 0:
            ta_tb_message = numpy.zeros(self.top_shape4)
            tb_ta_message = numpy.zeros(self.top_shape5)
        
        
        for i in range(0,self.bottom_batchsize):
            s_bot = bottom[0].data[i,0:self.slen].copy()
            s_bot_ed = numpy.reshape(s_bot,[self.nPeople+1, self.nScene])
            #print bottom[0].data[i,0:self.slen+400]
            #print s_bot_ed
            a_bot = bottom[0].data[i,self.slen:self.slen+self.alen].copy()
            a_bot_ed = numpy.reshape(a_bot,[self.alen/self.nAction, self.nAction])
            if self.K_>0:
                t_lf_bot = bottom[0].data[i,self.slen+self.alen:self.slen+self.alen+self.tlen_leaf].copy()
            if self.K_>1:
                t_md_bot = bottom[0].data[i,self.slen+self.alen+self.tlen_leaf:self.slen+self.alen+self.tlen_leaf+self.tlen_mid].copy()
                t_md_boted = numpy.reshape(t_md_bot,[self.tlen_mid/self.nAction, self.nAction])

            # scene to action            
            s_bot_ed[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nScene])
            if self.ifensemble:
                weighted_scene_unary = bottom_ms2_data[i,0:self.nScene]
                s_bot_ed[self.nPeople] = weighted_scene_unary
            unary_scene = s_bot_ed[self.nPeople].copy()
            s_bot_ed[self.nPeople] = numpy.zeros(self.nScene)
            tmp_all = s_bot_ed.sum(axis = 0)
            for j in range(0,self.nPeople):
                if j >= label_stop[i]:
                    s_a_message[count1] = numpy.zeros([1,self.nScene])
                    count1 += 1
                    continue
                tmp = tmp_all - s_bot_ed[j]
                #if self.ifprevious:
                #    tmp += scene_score_tms1[i]
                if self.ifnormalize ==True:
                    tmp = tmp/max(1,(1.0*(label_stop[i]-1)+1.0*self.ifprevious))
                    tmp += unary_scene
                    if self.ifprevious:
                        tmp += scene_score_tms1[i]
                        tmp = tmp/(3.0-(label_stop[i]==1))
                    else:
                        tmp = tmp/(2.0-(label_stop[i]==1))
                s_a_message[count1] = tmp.copy()
                count1 += 1
            # action to scene
            p_step = self.message_num_action
            for j in range(0,self.nPeople):
                if j >= label_stop[i]:
                    a_s_message[count2] = numpy.zeros([1,self.nAction])
                    count2 += 1
                    continue
                tmp_a = a_bot_ed[j*p_step:(j+1)*p_step].copy()
                shapetmp = tmp_a.shape
                tmp_a[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
                if self.ifensemble:
                    weighted_action_unary = bottom_ms2_data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction]
                    tmp_a[j] = weighted_action_unary
                unary_action = tmp_a[j].copy()
                tmp_a[j] = numpy.zeros(self.nAction)
                tmp_all = tmp_a.sum(axis = 0)
                tmp = tmp_all - tmp_a[self.nPeople]
                #if self.ifprevious: 
                #    tmp += action_score_tms1[i,j*self.nAction:(j+1)*self.nAction]
                if self.ifnormalize ==True:
                    tmp = tmp/max(1,((label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))
                    tmp += unary_action
                    if self.ifprevious:
                        tmp += action_score_tms1[i,j*self.nAction:(j+1)*self.nAction]
                        tmp = tmp/(3.0-(label_stop[i]==1))
                    else:
                        tmp = tmp/(2.0-(label_stop[i]==1))
                a_s_message[count2] = tmp.copy()
                count2 = count2 + 1
              
            # action to action
            p_step = self.message_num_action
            for p in range(0,self.nPeople):
                if p >= label_stop[i]:
                    a_a_message[count3:count3+self.nPeople] = numpy.zeros([self.nPeople,self.nAction])
                    count3+=self.nPeople
                    continue
                tmp_a = a_bot_ed[p*p_step:(p+1)*p_step].copy()
                tmp_a[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
                if self.ifensemble:
                    weighted_action_unary = bottom_ms1_data[i,j*self.nAction:(j+1)*self.nAction]
                    tmp_a[p] = weighted_action_unary
                shapetmp = tmp_a.shape
                unary_action = tmp_a[p] 
                tmp_a[p] = numpy.zeros(self.nAction)
                tmp_all = tmp_a.sum(axis = 0)
                for p_p in range(0,self.nPeople):
                    if p_p >= label_stop[i]:
                        a_a_message[count3] = numpy.zeros([1,self.nAction])
                        count3 = count3 + 1
                        continue
                    tmp = tmp_all - tmp_a[p_p]            # see how the messgeout layer handle the unary
                    #if self.ifprevious:
                    #    tmp += action_score_tms1_2[i,p*self.nAction:(p+1)*self.nAction]
                    if self.ifnormalize ==True:
                        tmp = tmp/max(1,(1.0*(label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))
                        tmp += unary_action
                        if self.ifprevious:
                            tmp += action_score_tms1_2[i,p*self.nAction:(p+1)*self.nAction]
                            tmp = tmp/(3.0-(label_stop[i]==1))
                        else:
                            tmp = tmp/(2.0-(label_stop[i]==1))
                    a_a_message[count3] = tmp.copy()
                    count3 = count3 + 1
            #print a_a_message
            if self.K_ < 1:
                continue
            # temporal after to temporal before
            midlen = self.tlen_mid/(2*(self.K_-1))
            for p in range(0,self.nPeople):
                if p>=label_stop[i]:
                    ta_tb_message[count4:count4+2*(self.K_)] = numpy.zeros([2*self.K_,self.nAction])
                    count4 += 2*self.K_
                    continue
                t_lf_bot_unit = t_lf_bot[p*2*self.nAction:(p+1)*2*self.nAction]
                # leaf_unit = t_lf_bot[0:2*self.nAction]
                ta_tb_message[count4] = t_lf_bot_unit[0:self.nAction].copy()
                count4 = count4 + 1
                md_step = self.nPeople*3
                for j in range(0,self.K_-1):
                    mid_unit = t_md_boted[j*md_step + p*3:j*md_step + (p+1)*3].copy()
                    tmp = mid_unit.sum(axis = 0)
                    tmp = (tmp - mid_unit[2])/2
                    ta_tb_message[count4] = tmp.copy()
                    count4 = count4 + 1
                tmp_a = a_bot_ed[p*p_step:(p+1)*p_step].copy()
                tmp_all = tmp_a.sum(axis = 0)
                shapetmp = tmp_a.shape
                tmp = tmp_all - tmp_a[shapetmp[0]]
                if self.ifnormalize ==True:
                    tmp = tmp/(1.0*shapetmp[0]-1.0)
                ta_tb_message[count4] = tmp.copy()
                count4 = count4 + 1
                for j in range(self.K_-1, 2*(self.K_-1)):
                    mid_unit = t_md_boted[j*md_step + p*3:j*md_step + (p+1)*3].copy()
                    tmp = mid_unit.sum(axis = 0)
                    tmp = (tmp - mid_unit[2])/2.0
                    ta_tb_message[count4] = tmp.copy()
                    count4 = count4 + 1

            # temporal before to temporal after
            midlen = self.tlen_mid/(2*(self.K_-1))
            stp = self.tlen_leaf/2
            for p in range(0,self.nPeople):
                if p >= label_stop[i]:
                    tb_ta_message[count5:count5+2*(self.K_)] = numpy.zeros([2*self.K_,self.nAction])
                    count5 += 2*self.K_
                    continue
                t_lf_bot_unit = t_lf_bot[stp+p*2*self.nAction:stp+(p+1)*2*self.nAction].copy()
                # leaf_unit = t_lf_bot_unit[2*self.nAction:4*self.nAction]
                tb_ta_message[count5] = t_lf_bot_unit[0:self.nAction].copy()
                count5 = count5 + 1
                md_step = 3*self.nPeople;
                theend = 2*(self.K_-1)-1
                for j in range(0,self.K_-1):
                    mid_unit = t_md_boted[(theend-j)*md_step + p*3:(theend-j)*md_step + (p+1)*3].copy()
                    tmp = mid_unit.sum(axis = 0)
                    if self.ifnormalize ==True:
                        tmp = (tmp - mid_unit[1])/2.0
                    tb_ta_message[count5] = tmp.copy()
                    count5 = count5 + 1
                tmp_a = a_bot_ed[p*p_step,(p+1)*p_step].copy()
                tmp_all = tmp_a.sum(axis = 0)
                shapetmp = tmp_a.shape
                tmp = tmp_all - tmp_a[shapetmp[0]-1]
                if self.ifnormalize ==True:
                    tmp = tmp/(1.0*shapetmp[0]-1.0)
                tb_ta_message[count5] = tmp.copy()
                count5 = count5 + 1
                for j in range(self.K_-1, 2*(self.K_-1)):
                    mid_unit = t_md_boted[(theend-j)*md_step + p*3:(theend-j)*md_step + (p+1)*3].copy()
                    tmp = mid_unit.sum(axis = 0)
                    if self.ifnormalize ==True:
                        tmp = (tmp - mid_unit[1])/2.0
                    tb_ta_message[count5] = tmp.copy()
                    count5 = count5 + 1
                
        top[0].data[...] = s_a_message.copy()
        top[1].data[...] = a_s_message.copy()
        top[2].data[...] = a_a_message.copy()
        if self.ifprevious == True:
            pass
            #print "data ---------------------------start"
            #print top[2].data[:5]
            #print "data ---------------------------end"
        if self.K_ > 0:
            top[3].data[...] = ta_tb_message.copy()
            top[4].data[...] = tb_ta_message.copy()

       

    def backward(self, top, propagate_down, bottom):
        #ifprevious shouldn't implement backprop -- not backprop to previous outputs
        #and add previous message emsemble to message_in layer!!!!!!!!!!!! 
        '''
        diff shapes:
        top_shape1 = [bottom_batchsize*self.nPeople, self.nScene]
        top_shape2 = [bottom_batchsize*self.nPeople, self.nAction]
        top_shape3 = [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        top_shape4 = [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        top_shape5 = [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        '''
        
        if len(bottom) <= 5:
            self.ifensemble = False;
        self.ifprevious = False
        if len(bottom) > 2 and self.prevhardgate:
            self.ifprevious = True
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize])
        labels = bottom[1].data
        for i in range(0,self.bottom_batchsize):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        if self.ifensemble:
            bottom_ms2_diff = numpy.zeros(bottom[-2].diff.shape)
            shape_ms1 = bottom[-1].diff.shape
            if shape_ms1[1] == self.nAction:
                bottom_ms1_diff = numpy.zeros([self.bottom_batchsize,self.nPeople*self.nAction])

        scene_diff = numpy.zeros([self.bottom_batchsize, self.slen])
        action_diff = numpy.zeros([self.bottom_batchsize, self.alen])
        if self.K_>0:
            temporal_leaf_diff = numpy.zeros([self.bottom_batchsize, self.tlen_leaf])
            temporal_mid_diff = numpy.zeros([self.bottom_batchsize, self.tlen_mid])

        count1 = 0;
        count2 = 0;
        count3 = 0;
        count3_1 = 2;
        count4_1 = 2;
        count3_2 = 0;
        count4_2 = 0;
        count3_3_3 = 1;
        count4_3_3 = 2*self.K_-1;
        count3_4_4 = 2*self.K_-1;
        count4_4_4 = 1;
        count5 = 0;
        label_stop = self.label_stop

        for i in range(0, self.bottom_batchsize):
            # collect diff on messages around scene node            
            for j in range(0, self.nPeople):
                if j>=label_stop[i]:
                    count1+=1
                    continue 
                tmp = numpy.repeat([top[0].diff[count1]], self.nPeople+1, axis = 0)
                tmp[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nScene])
                tmp[j] = numpy.zeros([1,self.nScene])       
                if self.ifnormalize ==True:                 
                    tmp = tmp/max((1.0*(label_stop[i]-1)+1.0*self.ifprevious),1)
                    tmp[self.nPeople] *= max((1.0*(label_stop[i]-1)+1.0*self.ifprevious),1)/(2.0-(label_stop[i]==1)+self.ifprevious)
                if self.ifprevious:
                    bottom[2].diff[i] += top[0].diff[count1]/(3.0-(label_stop[i]==1))
                if self.ifensemble:
                    bottom_ms2_diff[i,0:self.nScene] += top[0].diff[count1]/(1.0*label_stop[i]+1.0*self.ifprevious)
                    tmp[self.nPeople] = numpy.zeros([1,self.nScene])
                tmpshape = tmp.shape
                tmp = numpy.reshape(tmp,[1,tmpshape[0]*tmpshape[1]])
                scene_diff[i] = scene_diff[i] + tmp
                assert(count1%self.nPeople == j)
                count1 = count1+1

            # collect diff on messages around action nodes
            # from action nodes themselves
            tmpa_all = numpy.zeros([1,0])
            shape = self.nAction*(self.message_num_action)
            for p in range(0,self.nPeople):
                tmpa = numpy.zeros([1,shape])
                if p>=label_stop[i]:
                    count3+=self.nPeople
                    tmpa_all = numpy.append(tmpa_all,tmpa, axis = 1)
                    continue
                for p_p in range(0,self.nPeople):
                    if p_p >= label_stop[i]:
                        count3 += 1
                        continue
                    tmp = numpy.repeat([top[2].diff[count3]], self.message_num_action,axis= 0)
                    tmp[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
                    tmp[p_p] = numpy.zeros([1,self.nAction])
                    tmpshape = tmp.shape
                    if self.ifnormalize ==True:
                        tmp = tmp/max(1,(1.0*(label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))
                        tmp[p] *= max(1,(1.0*(label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))/(2.0-(label_stop[i]==1) + self.ifprevious)
                    if self.ifprevious:             
                        #bottom[3].diff[i,self.nAction*p:self.nAction*(p+1)] += top[2].diff[count3]/(1.0*label_stop[i]+2.0*(self.K_>0)+1.0*self.ifprevious) 
                        bottom[4].diff[i,self.nAction*p:self.nAction*(p+1)] += top[2].diff[count3]/(3.0-(label_stop[i]==1))
                    if self.ifensemble:
                        bottom_ms1_diff[i,p*self.nAction:(p+1)*self.nAction] += top[2].diff[count3]/(1.0*label_stop[i]+2.0*(self.K_>0)+1.0*self.ifprevious)
                        tmp[p] = numpy.zeros([1,self.nAction])
                    tmp = numpy.reshape(tmp,[1,shape])
                    count3 = count3+1      
                    tmpa = tmpa + tmp                  
                tmpa_all = numpy.append(tmpa_all,tmpa, axis = 1)
            length1 = tmpa_all.shape
            tmpa_all = numpy.reshape(tmpa_all, [self.nPeople, length1[1]/self.nPeople])
            # from scene nodes:
            shape = self.nAction*self.message_num_action
            for j in range(0,self.nPeople):
                if j >= label_stop[i]:
                    count2+=1
                    continue
                tmp = numpy.repeat([top[1].diff[count2]], self.message_num_action, axis = 0)
                tmp[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
                tmp[self.nPeople] = numpy.zeros([1,self.nAction])
                tmpsha = tmp.shape
                if self.ifnormalize ==True:
                    tmp = tmp/max(1,(1.0*(label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))
                    tmp *= max(1,(1.0*(label_stop[i]-1)+2.0*(self.K_>0)+1.0*self.ifprevious))/(2.0-(label_stop[i]==1)+self.ifprevious)
                if self.ifprevious:
                    bottom[3].diff[i,j*self.nAction:(j+1)*self.nAction] += top[1].diff[count2]/(3.0-(label_stop[i]==1))
                if self.ifensemble:
                    bottom_ms2_diff[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction] += top[1].diff[count2]/(1.0*label_stop[i]+2.0*(self.K_>0)+1.0*self.ifprevious)
                    tmp[j] = numpy.zeros([1,self.nAction])
                tmp = numpy.reshape(tmp,[1,shape])
                tmpa_all[j] = tmpa_all[j] + tmp
                count2 = count2+1
            if self.K_>=1:
                # from temporal nodes:
                position = self.K_
                shape = self.nAction*self.message_num_action
                for p in range(0, self.nPeople):
                    tmp = numpy.repeat([top[3].diff[count3_1]], self.message_num_action, axis = 0)
                    tmp[self.nPeople+2] = numpy.zeros([1,self.nAction])
                    tmpshape = tmp.shape
                    if self.ifnormalize ==True:
                        tmp = tmp/(1.0*tmpshape[0]-1.0)
                    tmp = numpy.reshape(tmp,[1,shape])
                    tmpa_all[p] = tmpa_all[p] + tmp
                    count3_1 = count3_1 + 1+2*(self.K_-1)+1 
                for p in range(0, self.nPeople):
                    tmp = numpy.repeat([top[4].diff[count4_1]], self.message_num_action, axis = 0)
                    tmp[self.nPeople+1] = numpy.zeros([1,self.nAction])
                    tmpshape = tmp.shape
                    if self.ifnormalize ==True:
                        tmp = tmp/(1.0*tmpshape[0]-1.0)
                    tmp = numpy.reshape(tmp,[1,shape])
                    tmpa_all[p] = tmpa_all[p] + tmp
                    count4_1 = count4_1 + 1+2*(self.K_-1)+1 
            length1 = tmpa_all.shape
            action_diff[i] = numpy.reshape(tmpa_all,[1,self.nPeople*length1[1]]).copy()
            #print action_diff
            if self.K_<1:
                continue    
            # collect diff on messages around temporal leaf nodes
            tmptlf_all = numpy.zeros([top[3].diff[count3_2].shape[0],0])
            for p in range(0,self.nPeople):
                tmp = top[3].diff[count3_2].copy()
                fortmp = numpy.zeros([1,self.nAction])
                tmp = numpy.append(tmp,fortmp,axis = 1)
                tmptlf_all = numpy.append(tmptlf_all,tmp,axis = 1)
                count3_2 = count3_2+1+2*(self.K_-1)+1
            for p in range(0,self.nPeople):
                tmp = top[4].diff[count4_2].copy()
                fortmp = numpy.zeros([1,self.nAction])
                tmp = numpy.append(tmp,fortmp,axis = 1)
                tmptlf_all = numpy.append(tmptlf_all,tmp,axis = 1)
                count4_2 = count4_2+1+2*(self.K_-1)+1
            temporal_leaf_diff[i] = tmptlf_all.copy()

            # collect diff on messages around temporal mid nodes
            tmpmd_all = numpy.zeros([1,0])
            for k in range(0,(self.K_-1)):
                count3_3 = count3_3_3 + k
                count4_3 = count4_3_3 + k
                tmp_b_all = numpy.zeros([self.nPeople,3*self.nAction])
                for p in range(0,self.nPeople):
                    tmp = numpy.repeat([top[3].diff[count3_3]],3,axis = 0)
                    tmp[2] = numpy.zeros([1,self.nAction])
                    tmp = numpy.reshape(tmp,[1,3*self.nAction])
                    tmp_b_all[p] = tmp_b_all[p]+tmp/2.0
                    tmp = numpy.repeat([top[4].diff[count4_3]],3,axis = 0)
                    tmp[1] = numpy.zeros([1,self.nAction])
                    tmp = numpy.reshape(tmp,[1,3*self.nAction])
                    tmp_b_all[p] = tmp_b_all[p]+tmp/2.0
                    count3_3 = count3_3+1+2*(self.K_-1)+1
                    count4_3 = count4_3+1+2*(self.K_-1)+1
                tmpmd_all = numpy.append(tmpmd_all,numpy.reshape(tmp_b_all,[1,self.nPeople*3*self.nAction]),axis=1)
            for k in range(self.K_-1,2*(self.K_-1)):
                count3_4 = count3_4_4 + k
                count4_4 = count4_4_4 + k
                tmp_b_all = numpy.zeros([self.nPeople,3*self.nAction])
                for p in range(0,self.nPeople):
                    tmp = numpy.repeat([top[3].diff[count3_4]],3,axis = 0)
                    tmp[2] = numpy.zeros([1,self.nAction])
                    tmp = numpy.reshape(tmp,[1,3*self.nAction])
                    tmp_b_all[p] = tmp_b_all[p]+tmp/2.0
                    tmp = numpy.repeat([top[4].diff[count4_4]],3,axis = 0)
                    tmp[1] = numpy.zeros([1,self.nAction])
                    tmp = numpy.reshape(tmp,[1,3*self.nAction])
                    tmp_b_all[p] = tmp_b_all[p]+tmp/2.0
                    count3_4 = count3_4+1+2*(self.K_-1)+1
                    count4_4 = count4_4+1+2*(self.K_-1)+1
                tmpmd_all = numpy.append(tmpmd_all,numpy.reshape(tmp_b_all,[1,self.nPeople*3*self.nAction]),axis = 1)
            temporal_mid_diff[i] = tmpmd_all.copy()   
        tmpall = scene_diff.copy()
        tmpall = numpy.append(tmpall,action_diff,axis = 1) 
        if self.K_>0:
            tmpall = numpy.append(tmpall,temporal_leaf_diff,axis = 1) 
        if self.K_>1:
            tmpall = numpy.append(tmpall,temporal_mid_diff,axis = 1)
        bottom[0].diff[...] =  tmpall.copy()
        if self.ifensemble:
            if bottom[-1].diff.shape[1] == self.nAction:
                bottom_ms1_diff = numpy.reshape(bottom_ms1_diff,bottom[-1].diff.shape)
            bottom[-1].diff[...] = bottom_ms1_diff
            bottom[-2].diff[...] = bottom_ms2_diff
        if self.ifprevious:
            pass#print bottom[2].diff
        '''if len(bottom) > 2:
            bottom[0].diff[...] =  0*tmpall
        else:
            bottom[0].diff[...] =  tmpall'''

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
