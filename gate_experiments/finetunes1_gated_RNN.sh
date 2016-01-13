#../caffe-master/build/tools/caffe train -weights  iter3_gated_snapshots_s1_iter_300_8075.caffemodel -solver=solver_finetune_gated_RNN.prototxt --gpu=0

../caffe-master/build/tools/caffe train -weights  iter1_fully_snapshots_s1_iter_10000_new.caffemodel -solver=solver_finetune_gated_RNN.prototxt --gpu=0

#../caffe-master/build/tools/caffe train -weights  new_exp/snapshots_iter2_fully/iter2_snapshots_s1_iter_12000.caffemodel -solver=solver_finetune_gated_RNN.prototxt --gpu=0
