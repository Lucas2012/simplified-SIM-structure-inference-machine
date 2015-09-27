import unittest
import tempfile
import os
import numpy
import caffe

class test_Message_Out(caffe.Layer):
    """A layer that just multiplies by ten"""

    def setup(self, bottom, top):
        self.nScene = 5
        self.nAction = 40
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
        tmpU = bottom[0].data[:,0:self.nScene].copy()
        #print bottom[0].data[:,:5]
        #print numpy.reshape(bottom[0].data[:,5:],[2,14,40])
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

                '''# unary term has fixed position in message vector:
                unary_action = bottom[0].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction]
                tmpj[1:j+1] = tmpj[0:j]
                tmpj[0] = unary_action
                #print unary_action'''
                # add unary input scores to the output messages:              
                unary_action = bottom[0].data[i,self.nScene+j*self.nAction:self.nScene+(j+1)*self.nAction].copy()
                tmpj[j] = unary_action
                #print unary_action

                #print tmpj[j]
                tmpj = numpy.reshape(tmpj,[1,self.nPeople*self.nAction])
                tmp1 = numpy.append(tmp1,tmpj,axis = 0).copy()

            #print numpy.reshape(tmp1,[14,14,40])
            #assert((0.025 in tmp1) == False)
            # scene to action
            tmpa_s = action_s[i].copy()
            #print tmp1.shape
            #print tmpa_s
            tmpa_s[label_stop[i]:self.nPeople] = numpy.zeros([self.nPeople-label_stop[i],self.nAction])
            #print tmpa_s
            #print tmp1.shape
            #print tmpa_s.shape
            #assert((0.025 in tmpa_s) == False)
            tmp1 = numpy.append(tmp1,tmpa_s,axis = 1).copy()
            #print tmp1.shape
            #print numpy.reshape(tmp1,[14,15,40])
            #assert((0.025 in tmp1) == False)
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
                    unary_action = action_pred[i,j,j*self.nAction:(j+1)*self.nAction].copy()
                    #print unary_action
                    action_pred[i,j,self.nAction:(j+1)*self.nAction] = action_pred[i,j,0:j*self.nAction].copy()
                    action_pred[i,j,0:self.nAction] = unary_action.copy()
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
        #print top[2].data[0]
        

    def backward(self, top, propagate_down, bottom):
        # TO DO: 717: 1. check if all the commented codes are useless.  2. complete the fixed unary option in the backward function
        # diffs from message_in_(k+1), scene_pred and action_pred
      
        #print "id",3-self.id%3
        self.id += 1
        label_stop = self.nPeople*numpy.ones([self.bottom_batchsize_frame])
        #print self.K_
        if self.K_>0:
            labels = bottom[8].data
        else:
            labels = bottom[6].data
        for i in range(0,self.bottom_batchsize_frame):
            #print i
            for j in range(0,self.nPeople):
                if labels[i*self.nPeople+j] == 0:
                    label_stop[i] = j
                    break
        label_stop = [1,2,3,4,14]
        self.label_stop = label_stop
        #print "cp1"
        
        diff0 = top[0].diff.copy()
        diff1 = top[1].diff.copy()
            
        action_pred_diff = top[2].diff.copy()   
        # change the fixed unaries back to ordered unary position
        for i in range(0,self.top_batchsize):
            print top[2].diff[i,1,0:40]
            for j in range(0,self.nPeople): 
                unary_unit = action_pred_diff[i,j,0:self.nAction].copy() 
                if j == -1:
                    print unary_unit
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
        print "in layer:",top[1].diff[1,0:5]
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
        print "in layer check:",top[2].diff[1,0,0:80]
        # probbly change the use of top[1,2,3] to see if it affects?    
        #print "New Batch:"                  
        for i in range(0,self.top_batchsize):            #--->set zeros to unary part
            #print "i",i
            # for bottom[1]:
            print "labelst",self.label_stop[i]
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
            if i >= 1:
                print "tmp2:",tmpdiff2[1,0:5]
            if tmpdiff2.shape[0] == 0:
                tmpdiff2 = tmp2.copy()
            else:
                tmpdiff2 = numpy.append(tmpdiff2,tmp2,axis = 0).copy()
            # bottom[3]:
            tmp2 = top[2].diff[i].copy()
            tmp2 = tmp2[:,0:self.nPeople*self.nAction]
            for p in range(0,self.nPeople):
                print self.label_stop[i],"check"
                if p >= self.label_stop[i]:
                    tmp3 = numpy.zeros([self.nPeople,self.nAction])
                else:
                    tmp3 = tmp2[:,p*self.nAction:(p+1)*self.nAction].copy()
                    tmp3[p] = numpy.zeros([1,self.nAction])
                    if i == 1 and p == 0:
                        print "cp:",tmp3[1]
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
                #print tmp.shape
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
        print bottom[3].diff[0,:]

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
        print "message out end"

def python_net_file():
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""name: 'pythonnet' force_backward: true
        input: 'data' input_shape { dim: 5 dim: 565 }
        input: 'label' input_shape {dim: 70}
        input: 'action_unary' input_shape { dim: 5 dim: 560}
        input: 'scene_unary' input_shape { dim: 5 dim: 5}
        input: 's2a' input_shape {dim: 70 dim: 40}
        input: 'a2s' input_shape {dim: 70 dim: 5}
        input: 'a2a' input_shape {dim: 980 dim: 40}
        layer { type: 'Python' name: 'one' bottom: 'data' bottom: 's2a' bottom: 'a2s' bottom: 'a2a' bottom: 'action_unary'  bottom: 'scene_unary' bottom: 'label' top: 'three' top: 'message1' top: 'message2' 
          python_param { module: 'test_Message_Out' layer: 'test_Message_Out' } }""")
        return f.name

class TestMessageOut(unittest.TestCase):
    def setUp(self):
        net_file = python_net_file()
        self.net = caffe.Net(net_file, caffe.TRAIN)
        os.remove(net_file)

    def test_forward(self):
        '''
        bottom[0] : unary input
        bottom[1] : scene->action   [bottom_batchsize*self.nPeople, self.nAction]
        bottom[2] : action->scene    [bottom_batchsize*self.nPeople, self.nScene]
        bottom[3] : action->action  [bottom_batchsize*self.nPeople*self.nPeople, self.nAction]
        bottom[4] : action prediction results from former iteration
        bottom[5] : frame prediction results from former iteration
        (optional)bottom[6] : temporal after->temporal before  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        (optional)bottom[7] : temporal before->temporal after  [bottom_batchsize*self.nPeople*2*self.K_, self.nAction]
        bottom[8] : action labels
        '''
        # test layer:
        # data input: data_x(5,5+14*40), label(70)
        # layer bottoms: s2a(5*14,40), a2s(5*14,5), a2a(5*14*14,40)
        # tops: message_in(5,8475),message1(5,15*5),message2(5,14,15*40)-->version of no temporal connections
        # add the rest input blobs for mesasge_out layer: action_unary(5,560), scene_unary(5,5)

        # should all consider dummies??????
        # initialize all the input blobs:
        
        x_data_unit = list(xrange(0,565))
        x_label = []
        #x_action_unary_unit = list(xrange(-560,0))
        x_s2a_unit = [1000]*40
        x_a2s_unit = [1500]*5 
        x_a2a_unit = [2000]*40 
        labelst = [1,2,3,4,14]

        # action labels in each frame
        for a in range(0,5):
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            x_label = x_label + tmp
		# input:
        x_data = []
        x_action_unary = []
        x_scene_unary = []
        x_s2a = []
        x_a2s = []
        x_a2a = []
        x_data_filtered = []
        for i in range(0,5):
            x_data.append(x_data_unit)
            x_data_filtered_unit = list(xrange(0,565))
            x_data_filtered_unit[5+40*labelst[i]:] = (14-labelst[i])*40*[0]
            x_data_filtered.append(x_data_filtered_unit)
            x_action_unary_unit = list(xrange(-565,-565+labelst[i]*40)) + [0]*40*(14-labelst[i])
            x_scene_unary_unit = list(xrange(-565,-560))
            #print len(x_scene_unary_unit)
            x_action_unary.append(x_action_unary_unit)
            x_scene_unary.append(x_scene_unary_unit)
        for i in range(0,70):
            x_s2a.append(x_s2a_unit)
            x_a2s.append(x_a2s_unit)
        for i in range(0,980):
            x_a2a.append(x_a2a_unit)

        # the ground truth positions and values for 3 final output blobs:
        y_three = []   # need to filter out dummies
        y_message1 = []   # (5,80)
        y_message2 = []   # (5,14,640)
        y_three_temp = []

        s2a = numpy.reshape(x_s2a,[5,14*40])
        a2s = numpy.reshape(x_a2s,[5,14*5])
        a2a = numpy.reshape(x_a2a,[5,14,14*40])
        # filters:
        for i in range(0,5):
            idx = labelst[i]
            s2a_unit = s2a[i].copy()
            a2s_unit = a2s[i].copy()
            a2a_unit = a2a[i] .copy()
            x_data_filtered_unit = x_data_filtered[i]
            x_scene_unary_unit = x_scene_unary[i]
            x_action_unary_unit = x_action_unary[i]
            x_action_unary_unit = numpy.reshape(x_action_unary_unit,[14,40]).copy()

            blank = []
            blank_temp = []
            # scene input messages arrangements:
            a2s_unit[5*labelst[i]:] = numpy.zeros([1,(14-labelst[i])*5])
            blank = numpy.append(blank,a2s_unit,axis = 1)
            blank_temp = numpy.append(blank_temp,a2s_unit,axis = 1)
            blank = numpy.append(blank,x_data_filtered_unit[0:5],axis = 1)
            blank_temp = numpy.append(blank_temp,x_data_filtered_unit[0:5],axis = 1)
            # #blank = numpy.append(blank,x_scene_unary_unit)
            # action input messages arrangements
            a2a_unit[:,labelst[i]*40:] = numpy.zeros([14,(14-labelst[i])*40])  
            a2a_unit[labelst[i]:,:] = numpy.zeros([14-labelst[i],14*40])
            s2a_unit = numpy.reshape(s2a_unit,[14,40])
            s2a_unit[labelst[i]:,:] = numpy.zeros([14-labelst[i],40])
            ## --add unaries for action scores
            x_data_action_tmp = numpy.reshape(x_data_filtered_unit[5:],[14,40])
            tmp_a2a = a2a_unit.copy()
            for j in range(0,14):
                tmp_a2a[j,j*40:j*40+40] = x_data_action_tmp[j]
            for j in range(0,14):
                a2a_unit[j,40:j*40+40] = a2a_unit[j,0:j*40]
                a2a_unit[j,0:40] = x_data_action_tmp[j]
            a2a_unit = numpy.append(a2a_unit,s2a_unit,axis = 1)
            tmp_a2a = numpy.append(tmp_a2a,s2a_unit,axis = 1)
            # #a2a_unit = numpy.append(a2a_unit,x_data_action_tmp, axis = 1)
            tmp = numpy.reshape(tmp_a2a,[1,14*15*40])
            blank = numpy.append(blank,tmp)
            y_three = numpy.append(y_three,blank,axis = 0)
            tmp_three = numpy.reshape(a2a_unit,[1,14*15*40])
            blank_temp = numpy.append(blank_temp,tmp_three)
            y_three_temp = numpy.append(y_three_temp,blank_temp,axis = 0)
        y_three = numpy.reshape(y_three,[5,8475])
        y_three_temp = numpy.reshape(y_three_temp,[5,8475])
        y_message1 = y_three[:,0:75]
        y_message2 = numpy.reshape(y_three_temp[:,75:],[5,14,600])

        # input blobs:
        self.net.blobs['data'].data[...] = x_data
        self.net.blobs['label'].data[...] = x_label
        self.net.blobs['s2a'].data[...] = x_s2a
        self.net.blobs['a2s'].data[...] = x_a2s
        self.net.blobs['a2a'].data[...] = x_a2a
        self.net.blobs['action_unary'].data[...] = x_action_unary
        self.net.blobs['scene_unary'].data[...] = x_scene_unary

        # check output:
        self.net.forward()
        framenum = 0
        for batch1,batch2 in zip(self.net.blobs['three'].data,y_three):
            #print framenum
            #framenum += 1
            pos = 0
            #print batch1[675:755]
            for x,y in zip(batch1,batch2):
                #print pos,x,y
                pos += 1
                self.assertEqual(x, y)

        for batch1,batch2 in zip(self.net.blobs['message1'].data,y_message1):
            for x,y in zip(batch1,batch2):
                self.assertEqual(x, y)

        for batch1,batch2 in zip(self.net.blobs['message2'].data,y_message2):
            for line1,line2 in zip(batch1,batch2):
                for x,y in zip(line1,line2):
                    self.assertEqual(x, y)

    def test_backward(self):
        # tops: message_in(5,8475), message1(5,15*5), message2(5,14,15*40)
        # bottoms: unary_data(5,565), s2a(5*14,40), a2s(5*14,40), a2a(14*14*5,40), action_pred(5,560), scene_pred(5,5)

        # mistake one: fixed unary is also set for "message_in" output, which is incorrect
        # warning one: previous prediction scores not used
        
        l = [];
        for a in range(0,5):
            tmp = [1]*(a+1) + [0]*(14-a-1)
            if a == 4:
                tmp = [1]*14
            l = l + tmp
        self.net.blobs['label'].data[...] = l
        labelst = [1,2,3,4,14]
        messagein_diff_unit = numpy.reshape(list(xrange(0,8475)),[1,8475])
        message1_diff_unit = numpy.reshape(list(xrange(10000,10075)),[1,75])
        message2_diff_unit = numpy.reshape(list(xrange(-14*15*40,0)),[14,600])
        messagein_diff = numpy.zeros([0,messagein_diff_unit.shape[1]])
        message1_diff = numpy.zeros([0,message1_diff_unit.shape[1]])
        message2_diff = numpy.zeros([0,message2_diff_unit.shape[1]])

        for i in range(0,5):
            messagein_diff = numpy.append(messagein_diff,messagein_diff_unit,axis = 0).copy()
            message1_diff = numpy.append(message1_diff,message1_diff_unit,axis = 0).copy()
            message2_diff = numpy.append(message2_diff,message2_diff_unit,axis = 0).copy()
        tmp_action_diff = numpy.reshape(message2_diff,[5,8400])
        assert(tmp_action_diff[2,0] == -8400)
        assert(tmp_action_diff[2,8399] == -1)
        messageall_diff = numpy.append(message1_diff,tmp_action_diff,axis = 1)
        messageall_diff += messagein_diff
        message1_diff = numpy.reshape(message1_diff,[5,75]).copy()
        message2_diff = numpy.reshape(message2_diff,[5,14,600]).copy()
        message2_diff_input = numpy.reshape(message2_diff,[5,14,600]).copy()

        # change fixed unaries:
        for i in range(0,5):
            for j in range(0,14):
                unary_score = message2_diff[i,j,0:40].copy()
                message2_diff[i,j,0:j*40] = message2_diff[i,j,40:(j+1)*40].copy()
                message2_diff[i,j,j*40:(j+1)*40] = unary_score.copy()
                #print message2_diff[i,j]
        messagein_diff = numpy.reshape(messagein_diff,[5,8475]).copy()
        tmp_action_diff = numpy.reshape(message2_diff,[5,8400]).copy()
        #print tmp_action_diff[1,0:100]
        assert(tmp_action_diff[2,0] == -8400)
        assert(tmp_action_diff[2,8399] == -1)
        messageall_diff = numpy.append(message1_diff,tmp_action_diff,axis = 1).copy()
        messageall_diff += messagein_diff

        # unary scene diff:
        unary_scene_diff = messageall_diff[:,70:75].copy()

        # unary action diff:
        tmp = numpy.reshape(messageall_diff[:,75:],[5,14,600]).copy()
        unary_action_diff = tmp[:,:,0:40].copy()
        unary_action_diff = numpy.reshape(unary_action_diff,[5,560]).copy()
        for i in range(0,5):
            for j in range(0,14):
                if j >= labelst[i]:
                    unary_action_diff[i,j*40:(j+1)*40] = numpy.zeros(40)
                else:
                    unary_action_diff[i,j*40:(j+1)*40] = tmp[i,j,j*40:(j+1)*40]
        # unary data diff:
        unary_data_diff = numpy.append(unary_scene_diff,unary_action_diff,axis = 1).copy()

        # s2a diff:
        tmp = numpy.reshape(messageall_diff[:,75:],[70,600]).copy()
        s2a_diff = tmp[:,560:600].copy()
        s2a_diff = numpy.reshape(s2a_diff,[5,560])
        for i in range(0,5):
            s2a_diff[i,labelst[i]*40:] = numpy.zeros((14-labelst[i])*40)
        s2a_diff = numpy.reshape(s2a_diff,[70,40])

        # a2s diff:
        a2s_diff = numpy.reshape(messageall_diff[:,0:70],[70,5]).copy()
        a2s_diff = numpy.reshape(a2s_diff,[5,70])
        for i in range(0,5):
            a2s_diff[i,labelst[i]*5:] = numpy.zeros((14-labelst[i])*5)
        a2s_diff = numpy.reshape(a2s_diff,[70,5])

        # a2a diff:          
        tmp = numpy.reshape(messageall_diff[:,75:],[5,14,600]).copy()
        tmp = tmp[:,:,:560].copy()
        for i in range(0,5):
            for j in range(0,14):
                if j >= labelst[i]:
                    tmp[i,j,:] = numpy.zeros(14*40)
                tmp[i,j,labelst[i]*40:] = numpy.zeros((14-labelst[i])*40)
                tmp[i,j,j*40:(j+1)*40] = numpy.zeros(40)
        a2a_diff = numpy.reshape(tmp,[980,40]).copy()
        # diff of predictions should all be zeros
        action_unary_diff = numpy.zeros([5,560])
        scene_unary_diff = numpy.zeros([5,5])
        self.net.blobs['three'].diff[...] = messagein_diff.copy()
        self.net.blobs['message1'].diff[...] = message1_diff.copy()
        self.net.blobs['message2'].diff[...] = message2_diff_input.copy()
        self.net.backward()

        # check: 'data','s2a,'a2s','a2a','action_unary','scene_unary'
        p11 = 0
        for batch1,batch2 in zip(self.net.blobs['data'].diff,unary_data_diff): 
            #print batch1[5+45]
            p11 += 1
            p22 = 0
            for x,y in zip(batch1,batch2):
                #print y
                p22 += 1
                self.assertEqual(y, x)
        for batch1,batch2 in zip(self.net.blobs['s2a'].diff,s2a_diff):
            for x,y in zip(batch1,batch2):
                self.assertEqual(y, x)
        p11 = 0    
        for batch1,batch2 in zip(self.net.blobs['a2s'].diff,a2s_diff):
            for x,y in zip(batch1,batch2):
                p22 += 1
                self.assertEqual(y, x)
        '''p11 = 0
        for batch1,batch2 in zip(self.net.blobs['a2a'].diff,a2a_diff):
            print p11
            if p11 == 196:
                print batch2
            p11 += 1
            p22 = 0
            for x,y in zip(batch1,batch2):
                #print p22
                p22 += 1

                self.assertEqual(y, x)'''

        for batch1,batch2 in zip(self.net.blobs['action_unary'].diff,action_unary_diff):
            for x,y in zip(batch1,batch2):
                self.assertEqual(y, x)

        for batch1,batch2 in zip(self.net.blobs['scene_unary'].diff,scene_unary_diff):
            for x,y in zip(batch1,batch2):
                self.assertEqual(y, x)

    def test_reshape(self):
        return 
        s = 4
        self.net.blobs['data'].reshape(s, s, s, s)
        self.net.forward()
        for blob in self.net.blobs.itervalues():
            for d in blob.data.shape:
                self.assertEqual(s, d)
