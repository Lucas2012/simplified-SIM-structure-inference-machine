import unittest
import tempfile
import os
import numpy
import caffe

class Message_Out(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 7
        self.nPeople = 14
        self.K_ = 0;
        self.top_batchsize = 0
        self.slen = 0
        self.alen = 0
        self.tlen_leaf = 0
        self.tlen_mid = 0 
        self.aunit = 0
        self.bottom_shape0 = [] # Unary input
        self.bottom_shape1 = [] # scene->action
        self.bottom_shape2 = [] # action->scene
        self.bottom_shape3 = [] # action->action
        self.bottom_shape4 = [] # temporal_after->temporal_before
        self.bottom_shape5 = [] # temporal_before->temporal_after
        self.message_num_action = self.nPeople+2*(self.K_>0)+1
        self.label_stop = []
        self.total = 0
        self.bottom_batchsize_frame = 0   # initial batchsize, for how many frames are used in one batch
        self.iffixedunary = True
        self.id = 0
        self.nounary = False
        self.ifensemble = False
        self.test_nonpairwise_action = False
    
    def reshape(self, bottom, top):
        # this layer has 6 input message blobs and 5 top blob output
        bottom1_shape = bottom[0].data.shape
        bottom1_batchsize = bottom1_shape[0]
        slen = (self.nPeople+1)*self.nScene
        alen = self.nPeople*self.message_num_action*self.nAction
        tlen_leaf = 0
        tlen_mid = 0
        top_batchsize = bottom1_batchsize
        top_shape = [top_batchsize, slen+alen+tlen_leaf+tlen_mid]
        self.total = top_shape[1]
        top[0].reshape(*top_shape)
        top[1].reshape(top_batchsize,(self.nPeople+1)*self.nScene)
        top[2].reshape(top_batchsize,self.nPeople,self.message_num_action*self.nAction)
        if self.K_ > 0:
            tlen_leaf = 2*self.nPeople*2*self.nAction # 2 leaves, n people, each has 2 input messages
            self.tlen_leaf = tlen_leaf
            top[3].reshape(self.top_batchsize,2,self.nPeople,2*self.nAction)
        if self.K_ > 1:
            tlen_mid = 2*(self.K_-1)*self.nPeople*3*self.nAction
            self.tlen_mid = tlen_mid
            top[4].reshape(self.top_batchsize,2*(self.K_-1),self.nPeople,3*self.nAction)
        
        self.slen = slen
        self.alen = alen
        self.top_batchsize = top_batchsize 
        self.bottom_batchsize_frame = bottom1_batchsize
        self.label_stop = [0]

    def forward(self, bottom, top):
        '''
        bottom[0] : unary input
        bottom[1] : scene->action   [bottom_batchsize*self.nPeople, self.nAction]
        bottom[2] : action->scene    [bottom_batchsize*self.nPeople, self.nScene]
        bottom[3] : action->action  [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        bottom[4] : temporal after->temporal before  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        bottom[5] : temporal before->temporal after  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        bottom[6] : action prediction results from former iteration
        bottom[7] : frame prediction results from former iteration
        # this layer has 5 diffs from top
        # top[0].diff for the whole message layer ---> [N,]
        # top[1].diff for scene prediction diff  ---> [N,(npeople+1)*nScene]
        # top[2].diff for action prediction diff  ---> [N,npeople,(npeople+2+1)*nAction]
        # top[3].diff for temporal_leaf prediction diff--->[N,2,npeople,2*nAction]
        # top[4].diff for temporal_mid prediction diff ---> [N,2*(self.K_-1),npeople,3*nAction]
        '''
        #print bottom[2].data[0]
        if self.K_>0:
            labels = bottom[8].data
        else:
            labels = bottom[6].data
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize_frame])
        mask_s = numpy.ones([self.top_batchsize,(self.nPeople+1)*self.nScene])
        for i in range(0,self.bottom_batchsize_frame):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    mask_s[i,label_stop[i]*self.nScene:self.nPeople*self.nScene] = numpy.zeros([1,(self.nPeople-label_stop[i])*self.nScene])
                    break
        self.label_stop = label_stop
        scenenode = numpy.zeros([0,0])
        # scene node:
        tmp = numpy.reshape(bottom[2].data,[self.top_batchsize,self.nPeople*self.nScene]).copy()
        #print tmp
        #print bottom[2].data
        tmpU = bottom[0].data[:,0:self.nScene].copy()
        #print bottom[0].data[:,:5]
        #print numpy.reshape(bottom[0].data[:,5:],[2,14,40])
        #print bottom[7].data
        scenenode = numpy.append(tmp,tmpU,axis = 1)
        scenenode = numpy.multiply(scenenode,mask_s)
        #print scenenode
        tmpa_all = numpy.reshape(bottom[3].data,[self.top_batchsize,self.nPeople,self.nPeople*self.nAction]).copy()
        actionnode = numpy.zeros([0,0])
        action_s = numpy.reshape(bottom[1].data,[self.top_batchsize,self.nPeople,self.nAction]).copy()
        #print bottom[1].data
        #print action_s
        if self.K_>0:
            ta_tb = numpy.reshape(bottom[4].data,[self.top_batchsize,self.nPeople,self.K_*2*self.nAction]).copy()
            tb_ta = numpy.reshape(bottom[5].data,[self.top_batchsize,self.nPeople,self.K_*2*self.nAction]).copy()
            temporal_lf = numpy.zeros([0,0])
        if self.K_>1:
            temporal_md = numpy.zeros([0,0])
            unit = self.tlen_mid/(2.0*self.K_-2.0)
        for i in range(0,self.top_batchsize):
            # action node:
            # action to action:
            assert(label_stop[i]>0)
            tmpa = tmpa_all[i].copy()
            tmp1 = numpy.zeros([0,self.nPeople*self.nAction])
            for j in range(0,self.nPeople):                #--- add unary
                if j >= label_stop[i]:
                    tmpj = numpy.zeros([1,self.nPeople*self.nAction])
                    tmp1 = numpy.append(tmp1,tmpj,axis = 0)
                    continue

                # temporarily set this to zeros, test if unaries are correct, remember to change it back!!!
                #tmpj = 0*tmpa[:,j*self.nAction:(j+1)*self.nAction]
                tmpj = tmpa[:,j*self.nAction:(j+1)*self.nAction].copy()             
                tmpj[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
                #print tmpj
                '''# unary term has fixed position in message vector:
                unary_action = bottom[0].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction]
                tmpj[1:j+1] = tmpj[0:j]
                tmpj[0] = unary_action
                #print unary_action'''
                # add unary input scores to the output messages:              
                unary_action = bottom[0].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction].copy()
                tmpj[j] = unary_action

                #print tmpj[j]
                tmpj = numpy.reshape(tmpj,[1,self.nPeople*self.nAction])
                tmp1 = numpy.append(tmp1,tmpj,axis = 0).copy()

            #assert((0.025 in tmp1) == False)
            # scene to action
            tmpa_s = action_s[i].copy()
            tmpa_s[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
            tmp1 = numpy.append(tmp1,tmpa_s,axis = 1).copy()
            if self.K_>=1:
                # t_after->t_before to action
                tmpab = ta_tb[i].copy()
                tmpa = tmpab[:,(self.K-1)*self.nAction,self.K_*self.nAction].copy()
                tmp1 = numpy.append(tmp1,tmpa,axis = 1).copy()
                # t_before->t_after to action     -----> what if only leaves! cannot handle k = 1
                tmpba = tb_ta[i].copy()
                tmpa = tmpba[:,(self.K-1)*self.nAction,self.K_*self.nAction].copy()
                tmp1 = numpy.append(tmp1,tmpa,axis = 1).copy()
                tmp1 = numpy.reshape(tmp1,[1,self.nPeople*self.message_num_action*self.nAction]).copy() # --->?
            #print tmp1
            if actionnode.shape[0] == 0:
                actionnode=tmp1.copy()
            else:
                actionnode = numpy.append(actionnode,tmp1,axis = 0).copy()
            if self.K_<1:
                continue
            #temporal nodes:
            #leaf                      
            tmp1 = numpy.zeros([0,0])      #-lack of unary
            stp = self.slen+self.alen
            tmpuse = bottom[0].data[i,stp:stp+self.tlen_leaf/2].copy()
            tmpuse = numpy.reshape(tmpuse,[self.nPeople,self.nAction])
            tmp1 = tmpuse
            tmpa = tmpba[:,(2*self.K_-1)*self.nAction:2*self.K_*self.nAction].copy()
            tmp11 = numpy.append(tmp1,tmpa,axis = 1).copy()
            # temporal_lf = numpy.append(temporal_lf,tmp1,axis = 0)
            tmp1 = numpy.zeros([0,0]) 
            tmpuse = bottom[0].data[i,stp+self.tlen_leaf/2:stp+self.tlen_leaf].copy()
            tmpuse = numpy.reshape(tmpuse,[self.nPeople,self.nAction])
            tmp1 = tmpuse
            tmpa = tmpab[:,(2*self.K_-1)*self.nAction:2*self.K_*self.nAction].copy()	
            tmp1 = numpy.append(tmp1,tmpa,axis = 1)
            tmp1 = numpy.append(tmp11,tmp1,axis = 0)
            tmp1 = numpy.reshape(tmp1,[1,2*self.nPeople*2*self.nAction])
            if temporal_lf.shape[0] == 0:
                temporal_lf = tmp1
            else:
                temporal_lf = numpy.append(temporal_lf,tmp1,axis = 0)
            #mid
            #tmpa = tmpab[:,0:self.nAction] - lack of unary
            stp = stp+self.tlen_leaf
            tm = numpy.zeros([1,0]) 
            for j in range(0,self.K_-1):
                tmp1 = numpy.zeros([0,0])
                tmpuse = bottom[0].data[i,stp:stp+unit].copy()
                tmpuse = numpy.reshape(tmpuse,[self.nPeople,self.nAction])
                tmpa = tmpab[:,j*self.nAction:(j+1)*self.nAction].copy()
                tmp1 = numpy.append(tmpuse,tmpa).copy()  # add axis!
                flag = 2*self.K_-2-j
                tmpa = tmpba[:,flag*self.nAction:(flag+1)*self.nAction].copy()
                tmp1 = numpy.append(tmp1,tmpa)      # add axis!
                tmp1 = numpy.reshape(tmp1,[1,self.nPeople*3*self.nAction])
                tm = numpy.append(tm,tmp1,axis = 1)
                stp = stp+unit
            for j in range(self.K_,2*self.K_-2):
                tmp1 = numpy.zeros([0,0])
                tmpuse = bottom[0].data[i,stp:stp+unit].copy()
                tmpuse = numpy.reshape(tmpuse,[self.nPeople,self.nAction])
                tmpa = tmpab[:,j*self.nAction:(j+1)*self.nAction].copy()
                tmp1 = numpy.append(tmpuse,tmpa).copy()     # add axis!
                flag = 2*self.K_-2-j
                tmpa = tmpba[:,flag*self.nAction,(flag+1)*self.nAction].copy()
                tmp1 = numpy.append(tmp1,tmpa)     # add axis!
                tmp1 = numpy.reshape(tmp1,[1,self.nPeople*3*self.nAction])
                tm = numpy.append(tm,tmp1,axis = 1)
                stp = stp+unit
            if temporal_md.shape[0] == 0:
                temporal_md = tm
            else:
                temporal_md = numpy.append(temporal_md,tm,axis = 0)

        tmp_all = scenenode.copy() 
        # scene node prediction input:
        top[1].data[...] = scenenode.copy()
        #print top[1].data
        if self.ifensemble:
            top[1].data[:,-self.nScene:] = bottom[7].data[:,0:self.nScene]
        # #scene predictions from previous iteration
        if self.K_>0:
            scenenode = numpy.append(scenenode,bottom[7].data,axis = 1).copy()
        else:
            scenenode = numpy.append(scenenode,bottom[5].data,axis = 1).copy()

        actionnode = numpy.reshape(actionnode,[self.top_batchsize,self.nPeople,self.message_num_action*self.nAction]).copy()

        # if using fixed position for unary scores:
        action_pred = actionnode.copy()
        if self.iffixedunary == True:
            for i in range(0,self.top_batchsize):
                for j in range(0,self.nPeople):
                    if j >=self.label_stop[i]:
                        action_pred[i,j,:] = numpy.zeros(len(action_pred[i,j,:]))
                    if self.ifensemble:
                        unary_action = bottom[7].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction]
                    else:
                        unary_action = action_pred[i,j,j*self.nAction:(j+1)*self.nAction].copy()
                    if self.nounary:
                        unary_action = numpy.zeros(self.nAction)
                    action_pred[i,j,self.nAction:(j+1)*self.nAction] = action_pred[i,j,0:j*self.nAction].copy()
                    action_pred[i,j,0:self.nAction] = unary_action.copy()
                    if self.test_nonpairwise_action:
                        action_pred[i,j,self.nAction:self.nPeople*self.nAction] = numpy.zeros((self.nPeople-1)*self.nAction)
                    if self.id%2 == 0:
                        pass#print action_pred[i,j,self.nAction:4*self.nAction]
                    if action_pred[i,j,self.nAction:self.nAction*2].all() != 0.0:
                        pass#print action_pred[i,j,self.nAction:self.nAction*2]
                    #print action_pred[i,j]
        top[2].data[...] = action_pred.copy()

        # temporal leaves:
        if self.K_>0:
            output_action = numpy.reshape(bottom[6].data,[self.top_batchsize,self.nPeople,self.nAction]).copy()
        else:
            output_action = numpy.reshape(bottom[4].data,[self.top_batchsize,self.nPeople,self.nAction]).copy()

        # action predictions from previous iteration
        aaa = numpy.append(actionnode,output_action,axis = 2).copy()

        # message_in_k+1 output:
        actionnode = numpy.reshape(actionnode,[self.top_batchsize,self.nPeople*self.message_num_action*self.nAction]).copy()
        tmp_all = numpy.append(tmp_all,actionnode,axis = 1) 
        if self.K_ > 0:
            tmp_all = numpy.append(tmp_all,temporal_lf,axis = 1).copy()
            top[3].data[...] = numpy.reshape(temporal_lf,[self.top_batchsize,2,self.nPeople,2*self.nAction]).copy()
        if self.K_>1:
            tmp_all = numpy.append(tmp_all,temporal_md,axis = 1).copy()   
            top[4].data[...] = numpy.reshape(temporal_md,[self.top_batchsize,2*(self.K_-1),self.nPeople,3*self.nAction]).copy()
        top[0].data[...] = tmp_all.copy()
        #print numpy.reshape(top[2].data[0],[14,15,40])
        #print top[0].data[0,0:75]
        #print top[2].data[0,1]
        #print "1:"
        #print top[1].data[0,1]  
        self.id += 1      

    def backward(self, top, propagate_down, bottom):
        # TO DO: 717: 1. check if all the commented codes are useless.  2. complete the fixed unary option in the backward function
        # diffs from message_in_(k+1), scene_pred and action_pred
      
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize_frame])
        if self.K_>0:
            labels = bottom[8].data
        else:
            labels = bottom[6].data
        for i in range(0,self.bottom_batchsize_frame):
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        self.label_stop = label_stop

        #print "id",3-self.id%3
        self.id += 1
        
        diff0 = top[0].diff.copy()
        diff1 = top[1].diff.copy()
            
        action_pred_diff = top[2].diff.copy()   
        if self.ifensemble:
            diff1[:,-self.nScene:] = numpy.zeros([self.top_batchsize,self.nScene])
        # change the fixed unaries back to ordered unary position
        for i in range(0,self.top_batchsize):
            for j in range(0,self.nPeople): 
                unary_unit = action_pred_diff[i,j,0:self.nAction].copy() 
                if j >= self.label_stop[i]:
                    action_pred_diff[i,j,:] = numpy.zeros(len(action_pred_diff[i,j,:])) 
                if self.ifensemble:
                    unary_unit = numpy.zeros(self.nAction)       
                action_pred_diff[i,j,0:j*self.nAction] = action_pred_diff[i,j,self.nAction:(j+1)*self.nAction].copy()
                action_pred_diff[i,j,j*self.nAction:(j+1)*self.nAction] = unary_unit.copy()
        action_pred_diff = numpy.reshape(action_pred_diff,[self.top_batchsize,self.nPeople*self.message_num_action*self.nAction]).copy()
        #print "check:"
        #print "shape:",top[1].diff.shape
        #print top[2].diff[1]
        diff1 = numpy.append(diff1,action_pred_diff, axis = 1).copy()
        if self.K_>0:
            action_pred_diff = numpy.reshape(top[3].diff,[self.top_batchsize,2*self.nPeople*2*self.nAction]).copy()
            diff1 = numpy.append(diff1,action_pred_diff, axis = 1).copy()
        if self.K_>1:
            action_pred_diff = numpy.reshape(top[4].diff,[self.top_batchsize,2*(self.K_-1)*npeople*3*nAction]).copy()
            diff1 = numpy.append(diff1,action_pred_diff, axis = 1).copy()
        # temporarily marked out this line to debug:
        diff = diff0+diff1
        stpp = self.slen
        top[1].diff[...] = diff[:,0:self.slen].copy()
        top[2].diff[...] = numpy.reshape(diff[:,stpp:stpp+self.alen],[self.top_batchsize,self.nPeople,self.message_num_action*self.nAction]).copy()
        #print top[2].diff[0]
        stpp+=self.alen
        if self.K_>0:
            top3diff = diff[:,stpp:stpp+self.tlen_leaf].copy()
            top[3].diff[...] = numpy.reshape(top3diff,[self.top_batchsize,2,2*self.nAction*self.nPeople]).copy()
            stpp+=self.tlen_leaf
        if self.K_>1:
            top4diff = diff[:,stpp:stpp+self.tlen_mid].copy()
            top[4].diff[...] = numpy.reshape(top4diff,[self.top_batchsize,2*(self.K_-1),3*self.nAction*self.nPeople]).copy()
        tmpdiff0 = numpy.zeros([0,0])
        tmpdiff1 = numpy.zeros([0,0])
        tmpdiff2 = numpy.zeros([0,0])
        tmpdiff3 = numpy.zeros([0,0])
        tmpdiff4 = numpy.zeros([0,0])
        tmpdiff5 = numpy.zeros([0,0])
        '''
        bottom[0] : unary input [bottom_batchsize,]
        bottom[1] : scene->action   [bottom_batchsize*self.nPeople, self.nAction]
        bottom[2] : action->scene    [bottom_batchsize*self.nPeople, self.nScene]
        bottom[3] : action->action  [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        bottom[4] : temporal after->temporal before  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        bottom[5] : temporal before->temporal after  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        '''
        # probbly change the use of top[1,2,3] to see if it affects?    
        #print "New Batch:"                  
        for i in range(0,self.top_batchsize):            #--->set zeros to unary part
            # for bottom[1]:
            tmp2 = top[2].diff[i].copy()
            
            tmp2 = tmp2[:,self.nPeople*self.nAction:(self.nPeople+1)*self.nAction].copy()
            
            tmp2[self.label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-self.label_stop[i],self.nAction])
            #print i,":"
            #print tmp2[0].shape
            #print tmp2[0]
            if tmpdiff1.shape[0] == 0:
                tmpdiff1 = tmp2.copy()
            else:
                tmpdiff1 = numpy.append(tmpdiff1,tmp2,axis = 0).copy()
            # bottom[2]:
            tmp2 = top[1].diff[i].copy()
            tmp2 = tmp2[0:self.nPeople*self.nScene]
            tmp2 = numpy.reshape(tmp2,[self.nPeople,self.nScene])
            tmp2[self.label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-self.label_stop[i],self.nScene])
            if tmpdiff2.shape[0] == 0:
                tmpdiff2 = tmp2
            else:
                tmpdiff2 = numpy.append(tmpdiff2,tmp2,axis = 0)
            # bottom[3]:
            tmp2 = top[2].diff[i].copy()
            tmp2 = tmp2[:,0:self.nPeople*self.nAction]
            for p in range(0,self.nPeople):
                if self.test_nonpairwise_action:
                    tmp3 = numpy.zeros([self.nPeople,self.nAction])
                else:
                    if p >= self.label_stop[i]:
                        tmp3 = numpy.zeros([self.nPeople,self.nAction])
                    else:
                        tmp3 = tmp2[:,p*self.nAction:(p+1)*self.nAction].copy()
                        tmp3[p] = numpy.zeros([1,self.nAction])
                        tmp3[self.label_stop[i]:,:] = numpy.zeros([self.nPeople-self.label_stop[i],self.nAction])
                if tmpdiff3.shape[0] == 0:
                    tmpdiff3 = tmp3.copy()
                else:
                    tmpdiff3 = numpy.append(tmpdiff3,tmp3,axis = 0).copy()

            if self.K_ >= 1:
                # bottom[4]:
                tmp3 = top[4].diff[i].copy()
                tmp2 = top[3].diff[i].copy()
                tmp1 = top[2].diff[i].copy()
                tmp1 = tmp1[:,(self.nPeople+1)*self.nAction:(self.nPeople+2)*self.nAction]
                mdunit = 3*self.nAction
                lfunit = 2*self.nAction
                #pos = self.nAction*(self.nPeople+1)
                for p in range(0,self.nPeople):
                    tmp3_1 = tmp3[0:self.K_-1,p*mdunit:(p+1)*mdunit].copy()
                    tmp00 = tmp3_1[:,self.nAction:2*self.nAction].copy()
                    # diff from action node:
                    tmp00 = numpy.append(tmp00,tmp1[p],axis = 0).copy()
                    # diff from before frames:
                    tmp3_2 = tmp3[self.K_-1:2*(self.K_-1),p*mdunit:(p+1)*mdunit].copy()
                    tmp3_2 = tmp3_2[:,self.nAction:2*self.nAction].copy()
                    tmp3_3 = tmp2[1,p*lfunit:(p+1)*lfunit].copy()
                    tmp00 = numpy.append(tmp00,tmp3_2,axis = 0).copy()
                    tmp00 = numpy.append(tmp00,tmp3_3[self.nAction:2*self.nAction],axis = 0).copy()
                    if tmpdiff4.shape[0] == 0:
                        tmpdiff4 = tmp00.copy()
                    else:
                        tmpdiff4 = numpy.append(tmpdiff4,tmp00).copy()     # add axis!
                # bottom[5]:
                tmp3 = numpy.flipud(tmp3)
                tmp1 = top[2].diff[i].copy()
                tmp1 = tmp1[:,(self.nPeople+2)*self.nAction:(self.nPeople+3)*self.nAction]
                #pos = self.nAction*(self.nPeople+1)
                for p in range(0,self.nPeople):
                    tmp3_1 = tmp3[0:self.K_-1,p*mdunit:(p+1)*mdunit].copy()
                    tmp00 = tmp3_1[:,2*self.nAction:3*self.nAction].copy()
                    # diff from action node:
                    tmp00 = numpy.append(tmp00,tmp1[p],axis = 0).copy()
                    # diff from after frames:
                    tmp3_2 = tmp3[self.K_-1:2*(self.K_-1),p*mdunit:(p+1)*mdunit].copy()
                    tmp3_2 = tmp3_2[:,self.nAction:2*self.nAction]
                    tmp3_3 = tmp2[0,p*lfunit:(p+1)*lfunit].copy()
                    tmp00 = numpy.append(tmp00,tmp3_2,axis = 0)
                    tmp00 = numpy.append(tmp00,tmp3_3[self.nAction:2*self.nAction],axis = 0) 
                    if tmpdiff5.shape[0] == 0:
                        tmpdiff5 = tmp00.copy()
                    else:
                        tmpdiff5 = numpy.append(tmpdiff5,tmp00).copy()         # add axis!
            # bottom[0]:
            if tmpdiff0.shape[0] == 0:
                scene_unary_tmp = diff[i,self.nPeople*self.nScene:(self.nPeople+1)*self.nScene].copy()
                tmpdiff0 = scene_unary_tmp.copy()
            else:
                scene_unary_tmp = diff[i,self.nPeople*self.nScene:(self.nPeople+1)*self.nScene].copy()
                tmpdiff0 = numpy.append(tmpdiff0,scene_unary_tmp,axis = 1).copy()   

            tmpuse = numpy.reshape(diff[i,self.slen:self.slen+self.alen],[self.nPeople,self.message_num_action*self.nAction]).copy()

            # unary terms have fixed position:
            for p in range(0,self.nPeople):
                if p >= self.label_stop[i]:
                    tmp = numpy.zeros(self.nAction)
                else:
                    tmp = tmpuse[p,p*self.nAction:(p+1)*self.nAction].copy()
                if self.nounary:
                    tmp = numpy.zeros(self.nAction)
                tmpdiff0 = numpy.append(tmpdiff0,tmp,axis = 1).copy()   

            step = self.slen+self.alen
            if self.K_>=1:
                tmpuse = diff[i,step:step+self.tlen_leaf].copy()
                tmpuse = numpy.reshape(tmpuse,[2*self.nPeople,2*self.nAction])
                tmpuse = numpy.reshape(tmpuse[:,0:self.nAction],[1,2*self.nPeople*self.nAction])
                tmpdiff0 = numpy.append(tmpdiff0,tmpuse,axis = 1).copy() 
            if self.K_ >= 2:
                step = step + self.tlen_leaf
                tmpuse = diff[i,step:step+self.tlen_mid].copy()
                tmpuse = numpy.reshape(tmpuse,[2*(self.K_-1)*self.nPeople,3*self.nAction]).copy()
                tmpuse = numpy.reshape(tmpuse[:,0:self.nAction],[1,2*(self.K_-1)*self.nPeople*self.nAction]).copy()
                tmpdiff0 = numpy.append(tmpdiff0,tmpuse,axis = 1).copy()         # add axis!
        tmpdiff0 = numpy.reshape(tmpdiff0,bottom[0].diff.shape).copy()
        bottom[0].diff[...] = tmpdiff0.copy()
        bottom[1].diff[...] = tmpdiff1.copy()
        bottom[2].diff[...] = tmpdiff2.copy()
        bottom[3].diff[...] = tmpdiff3.copy()
        #print bottom[1].diff[0]

        # if use previous prediction, remember to calculate corresponding diffs

        #print bottom[3].diff[1] # all zeros!!!!!
        #print diff[0]
        #print tmpdiff0[1]
        #print bottom[1].diff.shape
        #print bottom[1].diff
        #print top[2].diff
        if self.K_>0:
            bottom[4].diff[...] = tmpdiff4.copy()
            bottom[5].diff[...] = tmpdiff5.copy()

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
        self.net.setup()

    def test_forward(self):
        '''
        bottom[0] : unary input     [bottom_batchsize, self.nScene+self.nPeople*self.nAction]
        bottom[1] : scene->action   [bottom_batchsize*self.nPeople, self.nAction]
        bottom[2] : action->scene    [bottom_batchsize*self.nPeople, self.nScene]
        bottom[3] : action->action  [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        bottom[4] : action label'''
        #x = range()
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
