#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/sequence_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BP_RNNLayer<Dtype>::BP_RecurrentInputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "MessageIn0";
}

template <typename Dtype>
void BP_RNNLayer<Dtype>::BP_RecurrentOutputBlobNames(vector<string>* names) const {
  names->resize(1);
  (*names)[0] = "MessageIn" + this->int_to_str(this->T_);
}

template <typename Dtype>
void BP_RNNLayer<Dtype>::BP_OutputBlobNames(vector<string>* names) const {
  names->resize(2);
  (*names)[0] = "o_action";
  (*names)[1] = "o_scene";
}

template <typename Dtype>
void BP_RNNLayer<Dtype>::BP_FillUnrolledNet(NetParameter* net_param) const {
  const int num_output = this->layer_param_.bp_recurrent_param().num_output();
  CHECK_GT(num_output, 0) << "num_output must be positive";
  const FillerParameter& weight_filler =
      this->layer_param_.bp_recurrent_param().weight_filler();
  const FillerParameter& bias_filler =
      this->layer_param_.bp_recurrent_param().bias_filler();

  // Add generic LayerParameter's (without bottoms/tops) of layer types we'll
  // use to save redundant code.

  //int num_people = this->layer_param_.bp_recurrent_param().num_people();
  int scene_class = this->layer_param_.bp_recurrent_param().scene_class();
  int action_class = this->layer_param_.bp_recurrent_param().action_class();
  
  
/* temporarily not use,  these flags are for controlling connections between nodes
  int spp = this->layer_param_.bp_recurrent_param().spp();
  int tpp = this->layer_param_.bp_recurrent_param().tpp();
  int ps = this->layer_param_.bp_recurrent_param().ps();

  int temp_node_connection_start, temp_node_connection_mid, spat_node_connection, scene_node_connection;
  if (spp == 1){
      spat_node_connection = num_people + 1 + 1;  //spatial_connection = number of people + 1 temporal + 1 scene
  }
  else{
      spat_node_connection = 1 + 1;  //spatial_connection = 1 temporal + 1 scene
  }
  if (tpp == 1){
      temp_node_connection_start = 2;
      temp_node_connection_mid = 3;
  }
  else{
      temp_node_connection_start = 1; // only by itself, unary factor node
      temp_node_connection_mid = 1;
      std::cout << "without temporal connection!" << std::endl;
  }
  if (ps == 1){
      scene_node_connection = num_people + 1; //scene_connection = number of people + 1 scene unary factor node
  }
  else{
      scene_node_connection = 1;  // without person-scene interaction, scene_node_connection = 1 scene unary factor node itself
      std::cout << "without scene-person interaction!" << std::endl;
  }
  */
  // message classifier definition: **************************************************************************  


  int bp_num_output = 0;
  // temporal node message classifier, 1 for leaf nodes, 2 for mid nodes
  bp_num_output = action_class;  //TA->TB
  LayerParameter temporal1;
  temporal1.set_type("InnerProduct");
  temporal1.mutable_inner_product_param()->set_num_output(bp_num_output);
  temporal1.mutable_inner_product_param()->set_bias_term(true);
  temporal1.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  temporal1.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  temporal1.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter temporal2;   //TB->TA
  temporal2.set_type("InnerProduct");
  temporal2.mutable_inner_product_param()->set_num_output(bp_num_output);
  temporal2.mutable_inner_product_param()->set_bias_term(true);
  temporal2.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  temporal2.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  temporal2.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  // spatial node message classifier, 1 for output toward temporal nodes, 2 for output toward spatial nodes, 3 for output toward scene nodes
  bp_num_output = action_class;      // A->A
  LayerParameter spatial1;
  spatial1.set_type("InnerProduct");
  spatial1.mutable_inner_product_param()->set_num_output(bp_num_output);
  spatial1.mutable_inner_product_param()->set_bias_term(true);
  spatial1.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  spatial1.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  spatial1.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);
  
  LayerParameter spatial2;
  spatial2.set_type("InnerProduct");
  spatial2.mutable_inner_product_param()->set_num_output(bp_num_output);
  spatial2.mutable_inner_product_param()->set_bias_term(true);
  spatial2.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  spatial2.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  spatial2.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  bp_num_output = scene_class;      //A->S
  LayerParameter spatial3;
  spatial3.set_type("InnerProduct");
  spatial3.mutable_inner_product_param()->set_num_output(bp_num_output);
  spatial3.mutable_inner_product_param()->set_bias_term(true);
  spatial3.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  spatial3.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  spatial3.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  // scene node message classifier
  bp_num_output = action_class;      //S->A
  LayerParameter scene;
  scene.set_type("InnerProduct");
  scene.mutable_inner_product_param()->set_num_output(bp_num_output);
  scene.mutable_inner_product_param()->set_bias_term(true);
  scene.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  scene.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  scene.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);
  
  // scene node output
  bp_num_output = scene_class;      //S->A
  LayerParameter scene_lb;
  scene_lb.set_type("InnerProduct");
  scene_lb.mutable_inner_product_param()->set_num_output(bp_num_output);
  scene_lb.mutable_inner_product_param()->set_bias_term(true);
  scene_lb.mutable_inner_product_param()->set_axis(1);     // batchsize*num_related_messages_all*inputsize
  scene_lb.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  scene_lb.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  // spatial node message classifier, 1 for output toward temporal nodes, 2 for output toward spatial nodes, 3 for output toward scene nodes
  bp_num_output = action_class;      // A->A
  LayerParameter spatial1_lb;
  spatial1_lb.set_type("InnerProduct");
  spatial1_lb.mutable_inner_product_param()->set_num_output(bp_num_output);
  spatial1_lb.mutable_inner_product_param()->set_bias_term(true);
  spatial1_lb.mutable_inner_product_param()->set_axis(2);     // batchsize*num_related_messages_all*inputsize
  spatial1_lb.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  spatial1_lb.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  // structured gate:
  bp_num_output = 1;      // A->A
  LayerParameter structured_gate;
  structured_gate.set_type("InnerProduct");
  structured_gate.mutable_inner_product_param()->set_num_output(bp_num_output);
  structured_gate.mutable_inner_product_param()->set_bias_term(true);
  structured_gate.mutable_inner_product_param()->set_axis(1);     // batchsize*num_related_messages_all*inputsize
  structured_gate.mutable_inner_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);
  structured_gate.mutable_inner_product_param()->
      mutable_bias_filler()->CopyFrom(bias_filler);

  LayerParameter eliwise_weight;
  eliwise_weight.set_type("EliwiseProduct");
  //eliwise_weight.mutable_eliwise_product_param()->set_num_output(bp_num_output);
  //eliwise_weight.mutable_eliwise_product_param()->set_bias_term(true);
  eliwise_weight.mutable_eliwise_product_param()->
      mutable_weight_filler()->CopyFrom(weight_filler);

  // message classifier definition end. *******************************************************************************

  // prediction output layer
  LayerParameter scene_output_concat_layer;
  scene_output_concat_layer.set_name("o_concat_scene");
  scene_output_concat_layer.set_type("Concat");
  scene_output_concat_layer.add_top("o_scene");
  scene_output_concat_layer.mutable_concat_param()->set_axis(0);
  
  LayerParameter action_output_concat_layer;
  action_output_concat_layer.set_name("o_concat_action");
  action_output_concat_layer.set_type("Concat");
  action_output_concat_layer.add_top("o_action_forcheck");
  action_output_concat_layer.mutable_concat_param()->set_axis(0);

  LayerParameter tanh_param;
  tanh_param.set_type("ReLU");


  LayerParameter sum_param;
  sum_param.set_type("Eltwise");
  sum_param.mutable_eltwise_param()->set_operation(
      EltwiseParameter_EltwiseOp_SUM);
 
  bool lb = true;
  int timecontrol = 0;
  bool context = false;
  bool separate_initial_weight = true;
  float w_lr_mult = 10,b_lr_mult = 20; 
  float w_decay_mult = 10,b_decay_mult = 0;
  bool ifgate = false;
  bool ifforprint = true;
  bool ifneuron = false;
  bool if_ensemble_message = false;
  bool if_ensemble_previous = false;
  

  // BP recurrent steps start
  // split frame scores and action scores
  {
	  LayerParameter* split_action_frame = net_param->add_layer();
      split_action_frame->set_name("slice_action_frame_0");
	  split_action_frame->set_type("Slice");
      split_action_frame->add_bottom("concat_all");
      split_action_frame->add_top("scene_score_normalized0");
      split_action_frame->add_top("cur_action_score_normalized_reshaped0");
      split_action_frame->mutable_slice_param()->add_slice_point(5);
  }
  

  LayerParameter action_output_python;
  action_output_python.set_name("output_python_concat_action");
  action_output_python.set_type("Python");
  action_output_python.mutable_python_param()->set_module("MyConcat");
  action_output_python.mutable_python_param()->set_layer("MyConcat");
  action_output_python.add_top("o_action_forcheck");

  LayerParameter scene_output_python;
  scene_output_python.set_name("output_python_concat_scene");
  scene_output_python.set_type("Python");
  scene_output_python.mutable_python_param()->set_module("MyConcat");
  scene_output_python.mutable_python_param()->set_layer("MyConcat");
  if (ifforprint){
      scene_output_python.add_top("scene_pred");
  }else{
      scene_output_python.add_top("o_scene");
  }  
  

  for (int t = 1; t <= this->T_; ++t) {
      string tm1s = this->int_to_str(t - 1);
      string ts = this->int_to_str(t);
      
      /*if (if_ensemble_previous && t > 1){
		  {
			  LayerParameter* previous_prediction_ensem_scene = net_param->add_layer();
			  previous_prediction_ensem_scene->CopyFrom(eliwise_weight);
			  previous_prediction_ensem_scene->set_name("previous_scene_ensemble" + tm1s);
		      caffe::ParamSpec* weight = previous_prediction_ensem_scene->add_param();
		      //caffe::ParamSpec* weight_bias = previous_prediction_ensem_scene->add_param();
		      weight->set_name("W_ps_1");
		      weight->set_lr_mult(w_lr_mult);
		      weight->set_decay_mult(w_decay_mult);
		      //weight_bias->set_lr_mult(b_lr_mult);
		      //weight_bias->set_decay_mult(b_decay_mult);
			  previous_prediction_ensem_scene->add_bottom("scene_score_normalized" + tm1s);
			  previous_prediction_ensem_scene->add_top("scene_score_normalized_weighted" + tm1s);          
			  //previous_prediction_ensem_scene->mutable_eliwise_product_param()->set_axis(1);  // batchsize*num_people
		  }
		  {
			  LayerParameter* previous_prediction_ensem_action = net_param->add_layer();
			  previous_prediction_ensem_action->CopyFrom(eliwise_weight);
			  previous_prediction_ensem_action->set_name("previous_action_ensemble" + tm1s);
		      caffe::ParamSpec* weight = previous_prediction_ensem_action->add_param();
		      //caffe::ParamSpec* weight_bias = previous_prediction_ensem_action->add_param();
		      weight->set_name("W_pa_1");
		      weight->set_lr_mult(w_lr_mult);
		      weight->set_decay_mult(w_decay_mult);
		      //weight_bias->set_lr_mult(b_lr_mult);
		      //weight_bias->set_decay_mult(b_decay_mult);
			  previous_prediction_ensem_action->add_bottom("cur_action_score" + tm1s);
			  previous_prediction_ensem_action->add_top("cur_action_score_weighted" + tm1s);          
			  //previous_prediction_ensem_action->mutable_eliwise_product_param()->set_axis(2);  // batchsize*num_people
		  }
      }*/
      
      if (t > 1){
          {
		      LayerParameter* graphical_edge = net_param->add_layer();
		      graphical_edge->set_name("graphical_edge" + ts);
			  graphical_edge->set_type("Python");
			  graphical_edge->mutable_python_param()->set_module("graphical_edge");
			  graphical_edge->mutable_python_param()->set_layer("graphical_edge");
			  if (ifforprint){
                  graphical_edge->add_bottom("concat_all");
              }
              else{
                  graphical_edge->add_bottom("UnaryInput");
              }
		      graphical_edge->add_bottom("MessageIn" + tm1s);
		      graphical_edge->add_bottom("label_action");
			  graphical_edge->add_top("gate_input" + ts);
          }
      

		  {
		      LayerParameter* gate_compute = net_param->add_layer();
		      gate_compute->CopyFrom(structured_gate);
		      gate_compute->set_name("gate_compute" + ts);
              caffe::ParamSpec* weight = gate_compute->add_param();
              caffe::ParamSpec* weight_bias = gate_compute->add_param();
              weight->set_name("W_gh_1");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
		      //gate_compute->add_param()->set_name("W_gh_1");
		      //gate_compute->add_param()->set_lr_mult(w_lr_mult);
		      //gate_compute->add_param()->set_decay_mult(w_decay_mult);
		      gate_compute->add_bottom("gate_input" + ts);
		      gate_compute->add_top("gates" + ts);          
		      gate_compute->mutable_inner_product_param()->set_axis(1);  // batchsize*num_people
		  }

          {
		      LayerParameter* structured_gate = net_param->add_layer();
		      structured_gate->set_name("structured_gate" + ts);
			  structured_gate->set_type("Python");
			  structured_gate->mutable_python_param()->set_module("structured_gate");
			  structured_gate->mutable_python_param()->set_layer("structured_gate");
			  structured_gate->add_bottom("gates" + ts);
		      structured_gate->add_bottom("MessageIn" + tm1s);
		      structured_gate->add_bottom("label_action");
			  structured_gate->add_top("gated_MessageIn" + tm1s);
          }
          
          /*{
              LayerParameter* tanh_gate = net_param->add_layer();
              tanh_gate->set_type("TanH");
              tanh_gate->add_bottom("gated_MessageIn_net" + tm1s);
              tanh_gate->add_top("gated_MessageIn" + tm1s);
          }*/
      }

      
      // Add layer to re-arrange all messages to each innerproduct component
      {
		  LayerParameter* python_message_in = net_param->add_layer();
          python_message_in->set_name("ArrangeMessageIn" + ts);
		  python_message_in->set_type("Python");
		  python_message_in->mutable_python_param()->set_module("Message_In");
		  python_message_in->mutable_python_param()->set_layer("Message_In");
          if (t > 1){
              if (ifgate){
		          python_message_in->add_bottom("gated_MessageIn" + tm1s);
              }
              else{
                  python_message_in->add_bottom("MessageIn" + tm1s);
              }
          }
          else{
              if (ifforprint){
		          python_message_in->add_bottom("Initial_Messages");
              }else{
                  python_message_in->add_bottom("MessageIn" + tm1s);
              }
          }
          python_message_in->add_bottom("label_action");
          if (t > 1){
              if (if_ensemble_previous){
                  python_message_in->add_bottom("scene_score_normalized_weighted" + tm1s);
                  python_message_in->add_bottom("cur_action_score_weighted1" + tm1s);
                  python_message_in->add_bottom("cur_action_score_weighted2" + tm1s);
              }else{
                  python_message_in->add_bottom("scene_score_normalized_blocked" + tm1s);
                  python_message_in->add_bottom("cur_action_score_blocked" + tm1s);
                  python_message_in->add_bottom("cur_action_score_blocked" + tm1s);
              }
              if (if_ensemble_message) {
                  python_message_in->add_bottom("concat_all_weighted");
                  python_message_in->add_bottom("fc8_CAD_prob_weighted2");
              }
          }
		  python_message_in->add_top("S_A_MessageIn" + ts);
		  python_message_in->add_top("A_S_MessageIn" + ts);
		  python_message_in->add_top("A_A_MessageIn" + ts);
          if (this->K_>0){
              python_message_in->add_top("Ta_Tb_MessageIn" + ts);
              python_message_in->add_top("Tb_Ta_MessageIn" + ts);
          }
      }

      // Add layer to predict messages toward action node from scene
      {
          LayerParameter* scene_message_pred = net_param->add_layer();
          scene_message_pred->CopyFrom(scene);
          scene_message_pred->set_name("S_A_Message" + ts);
          caffe::ParamSpec* weight = scene_message_pred->add_param();
          caffe::ParamSpec* weight_bias = scene_message_pred->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_hh_1" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_hh_1");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_hh_1");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          scene_message_pred->add_bottom("S_A_MessageIn" + ts);
          if (ifneuron){
              scene_message_pred->add_top("S_A_MessageOut_neuron_input_" + ts); 
          }else{
              scene_message_pred->add_top("S_A_MessageOut" + ts); 
          }         
          scene_message_pred->mutable_inner_product_param()->set_axis(1);  // batchsize*num_people
      }

      // Add layer to predict messages toward scene node from action
      {
          LayerParameter* scene_message_pred_as = net_param->add_layer();
          scene_message_pred_as->CopyFrom(spatial3);
          scene_message_pred_as->set_name("A_S_Message" + ts); 
          caffe::ParamSpec* weight = scene_message_pred_as->add_param();
          caffe::ParamSpec* weight_bias = scene_message_pred_as->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_hh_2" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_hh_2");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_hh_2");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          scene_message_pred_as->add_bottom("A_S_MessageIn" + ts);
          if (ifneuron){
              scene_message_pred_as->add_top("A_S_MessageOut_neuron_input_" + ts);  
          }else{
              scene_message_pred_as->add_top("A_S_MessageOut" + ts);
          }        
          scene_message_pred_as->mutable_inner_product_param()->set_axis(1);  // batchsize*num_people
      }
      
      // Add layer to predict messages toward action node under current frame
      {
          LayerParameter* action_message_pred = net_param->add_layer();
          action_message_pred->CopyFrom(spatial1);
          action_message_pred->set_name("A_A_Message" + ts);
          caffe::ParamSpec* weight = action_message_pred->add_param();
          caffe::ParamSpec* weight_bias = action_message_pred->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_hh_3" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_hh_3");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_hh_3");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          action_message_pred->add_bottom("A_A_MessageIn" + ts);
          if (ifneuron){
              action_message_pred->add_top("A_A_MessageOut_neuron_input_" + ts);  
          }else{
              action_message_pred->add_top("A_A_MessageOut" + ts); 
          }        
          action_message_pred->mutable_inner_product_param()->set_axis(1); // batchsize*num_people*num_people --->
      }

      // Add layer to predict temporal messages toward action node under before/after K frames, K is set to be 1 or 2
      if (this->K_>0){
      {
          LayerParameter* temporal_message_pred_ab = net_param->add_layer();
          temporal_message_pred_ab->CopyFrom(temporal1);
          temporal_message_pred_ab->set_name("Ta_Tb_Message" + ts);
          caffe::ParamSpec* weight = temporal_message_pred_ab->add_param();
          caffe::ParamSpec* weight_bias = temporal_message_pred_ab->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_hh_4" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_hh_4");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_hh_4");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          temporal_message_pred_ab->add_bottom("Ta_Tb_MessageIn" + ts);
          if (ifneuron){
              temporal_message_pred_ab->add_top("Ta_Tb_MessageOut_neuron_input_" + ts);  
          }else{
              temporal_message_pred_ab->add_top("Ta_Tb_MessageOut" + ts);
          }        
          temporal_message_pred_ab->mutable_inner_product_param()->set_axis(1); // batchsize*num_people*num_people --->
      }

      // Add layer to predict temporal messages toward action node under before/after K frames, K is set to be 1 or 2
      {
          LayerParameter* temporal_message_pred_ba = net_param->add_layer();
          temporal_message_pred_ba->CopyFrom(temporal1);
          temporal_message_pred_ba->set_name("Tb_Ta_Message" + ts);
          caffe::ParamSpec* weight = temporal_message_pred_ba->add_param();
          caffe::ParamSpec* weight_bias = temporal_message_pred_ba->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_hh_5" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_hh_5");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_hh_5");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          temporal_message_pred_ba->add_bottom("Tb_Ta_MessageIn" + ts);
          if (ifneuron){
              temporal_message_pred_ba->add_top("Tb_Ta_MessageOut_neuron_input_" + ts); 
          }else{
              temporal_message_pred_ba->add_top("Tb_Ta_MessageOut" + ts); 
          }         
          temporal_message_pred_ba->mutable_inner_product_param()->set_axis(1); // batchsize*num_people*num_people --->
      }
      }
      // neuron layers:
      if (ifneuron){
      {
          {
		    LayerParameter* h_neuron_param = net_param->add_layer();
		    h_neuron_param->CopyFrom(tanh_param);
		    h_neuron_param->set_name("h_neuron1_" + ts);
		    h_neuron_param->add_bottom("S_A_MessageOut_neuron_input_" + ts);
		    h_neuron_param->add_top("S_A_MessageOut" + ts);
		  }
          {
		    LayerParameter* h_neuron_param = net_param->add_layer();
		    h_neuron_param->CopyFrom(tanh_param);
		    h_neuron_param->set_name("h_neuron2_" + ts);
		    h_neuron_param->add_bottom("A_S_MessageOut_neuron_input_" + ts);
		    h_neuron_param->add_top("A_S_MessageOut" + ts);
		  }
          {
		    LayerParameter* h_neuron_param = net_param->add_layer();
		    h_neuron_param->CopyFrom(tanh_param);
		    h_neuron_param->set_name("h_neuron3_" + ts);
		    h_neuron_param->add_bottom("A_A_MessageOut_neuron_input_" + ts);
		    h_neuron_param->add_top("A_A_MessageOut" + ts);
		  }
      }
      }
      // Add layer to normalize message
      {
          LayerParameter* softmax_layer_normalize1 = net_param->add_layer();
          softmax_layer_normalize1->set_type("Softmax");
          softmax_layer_normalize1->set_name("Normalize_Scene_to_ActionM" + ts);
          softmax_layer_normalize1->add_bottom("S_A_MessageOut" + ts);
          softmax_layer_normalize1->add_top("S_A_MessageOut_normalized" + ts);
      }

      // Add layer to normalize message
      {
          LayerParameter* softmax_layer_normalize2 = net_param->add_layer();
          softmax_layer_normalize2->set_type("Softmax");
          softmax_layer_normalize2->set_name("Normalize_Action_to_SceneM" + ts);
          softmax_layer_normalize2->add_bottom("A_S_MessageOut" + ts);
          softmax_layer_normalize2->add_top("A_S_MessageOut_normalized" + ts);
      }
      
      // Add layer to normalize message
      {
          LayerParameter* softmax_layer_normalize3 = net_param->add_layer();
          softmax_layer_normalize3->set_type("Softmax");
          softmax_layer_normalize3->set_name("Normalize_Action_to_ActionM" + ts);
          softmax_layer_normalize3->add_bottom("A_A_MessageOut" + ts);
          softmax_layer_normalize3->add_top("A_A_MessageOut_normalized" + ts);
      }
      if (this->K_>0){
		  // Add layer to normalize message
		  {
		      LayerParameter* softmax_layer_normalize2 = net_param->add_layer();
		      softmax_layer_normalize2->set_type("Softmax");
		      softmax_layer_normalize2->set_name("Normalize_ActionM" + ts);
		      softmax_layer_normalize2->add_bottom("Ta_Tb_MessageOut" + ts);
		      softmax_layer_normalize2->add_top("Ta_Tb_MessageOut_normalized" + ts);
		  }
		  
		  // Add layer to normalize message
		  {
		      LayerParameter* softmax_layer_normalize3 = net_param->add_layer();
		      softmax_layer_normalize3->set_type("Softmax");
		      softmax_layer_normalize3->set_name("Normalize_TemporalM" + ts);
		      softmax_layer_normalize3->add_bottom("Tb_Ta_MessageOut" + ts);
		      softmax_layer_normalize3->add_top("Tb_Ta_MessageOut_normalized" + ts);
		  }
      }
      // Add layer to re-arrange messages from each component
      {
          LayerParameter* python_message_out = net_param->add_layer();
          python_message_out->set_name("ArrangeMessageOut" + ts);
          python_message_out->set_type("Python");
          python_message_out->mutable_python_param()->set_module("Message_Out");
          python_message_out->mutable_python_param()->set_layer("Message_Out");
          if (ifforprint){
              /*if (if_ensemble_message){
				  python_message_out->add_bottom("concat_all_weighted"); // change
			  }else{
				  python_message_out->add_bottom("concat_all");
			  }*/
              python_message_out->add_bottom("concat_all");
          }else{
              python_message_out->add_bottom("UnaryInput");
          }
          python_message_out->add_bottom("S_A_MessageOut_normalized" + ts);
          python_message_out->add_bottom("A_S_MessageOut_normalized" + ts);
          python_message_out->add_bottom("A_A_MessageOut_normalized" + ts);
          python_message_out->add_bottom("cur_action_score_normalized_reshaped" + tm1s);
          python_message_out->add_bottom("scene_score_normalized" + tm1s);

          if (this->K_>0){
              python_message_out->add_bottom("Ta_Tb_MessageOut_normalized" + ts);
              python_message_out->add_bottom("Tb_Ta_MessageOut_normalized" + ts);
          }
          python_message_out->add_bottom("label_action");
          //python_message_out->add_bottom("concat_all");
          python_message_out->add_top("MessageIn" + ts);
          python_message_out->add_top("Message_1" + ts);
          python_message_out->add_top("Message_2" + ts);
          if (this->K_>0){
		      python_message_out->add_top("Message_3" + ts);
		      python_message_out->add_top("Message_4" + ts);
          }
      }
      if (lb){
      // Add latent boost layer: scene node prediction
      {
          LayerParameter* Latent_Boost_Scene = net_param->add_layer();
          Latent_Boost_Scene->CopyFrom(scene_lb);
          Latent_Boost_Scene->set_name("LB_scene" + ts);
          caffe::ParamSpec* weight = Latent_Boost_Scene->add_param();
          caffe::ParamSpec* weight_bias = Latent_Boost_Scene->add_param();
          /*if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_ho_1" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_ho_1");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }*/
          // else{
              weight->set_name("W_ho_1");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          // }
          
          Latent_Boost_Scene->add_bottom("Message_1" + ts);
          //Latent_Boost_Scene->add_top("scene_score_neuron_input_" + ts);
          Latent_Boost_Scene->add_top("scene_score" + ts);
          Latent_Boost_Scene->mutable_inner_product_param()->set_axis(1); //---------------------- set axis
      }

      // Add layer to block propagation
      {
          LayerParameter* softmax_layer_normalize_scene = net_param->add_layer();
          softmax_layer_normalize_scene->set_type("Softmax");
          softmax_layer_normalize_scene->set_name("Normalize_Scene_Pred" + ts);
          softmax_layer_normalize_scene->add_bottom("scene_score" + ts);
          softmax_layer_normalize_scene->add_top("scene_score_normalized" + ts);
          softmax_layer_normalize_scene->mutable_softmax_param()->set_axis(1);
      }
      if (t!=this->T_){    
          {
              LayerParameter* python_block = net_param->add_layer();
              python_block->set_name("python_scene_block_prop" + ts);
              python_block->set_type("Python");
              python_block->mutable_python_param()->set_module("block_prop");
              python_block->mutable_python_param()->set_layer("block_prop");
              python_block->add_bottom("scene_score_normalized" + ts);
              python_block->add_top("scene_score_normalized_blocked" + ts);
          }
          {
              LayerParameter* eliwiseproduct_layer = net_param->add_layer();
              eliwiseproduct_layer->CopyFrom(eliwise_weight);
              eliwiseproduct_layer->set_name("weight_scene_prev" + ts);
              caffe::ParamSpec* weight = eliwiseproduct_layer->add_param();
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              eliwiseproduct_layer->add_bottom("scene_score_normalized_blocked" + ts);
              eliwiseproduct_layer->add_top("scene_score_normalized_weighted" + ts);
          }
      }

      // Add latent boost layer: Current Frame action node prediction
      {
          LayerParameter* Latent_Boost_Action = net_param->add_layer();
          Latent_Boost_Action->CopyFrom(spatial1_lb);
          Latent_Boost_Action->set_name("LB_action" + ts);
          caffe::ParamSpec* weight = Latent_Boost_Action->add_param();
          caffe::ParamSpec* weight_bias = Latent_Boost_Action->add_param();
          /*if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_ho_2" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_ho_2");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }*/
          //else{
              weight->set_name("W_ho_2");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          //}
          Latent_Boost_Action->add_bottom("Message_2" + ts);
          //Latent_Boost_Action->add_top("cur_action_score_neuron_input_" + ts);
          Latent_Boost_Action->add_top("cur_action_score" + ts);
          Latent_Boost_Action->mutable_inner_product_param()->set_axis(2); //---------------------- set axis
      }
      if (t!=this->T_){
          {
              LayerParameter* python_block = net_param->add_layer();
              python_block->set_type("Python");
              python_block->set_name("python_action_block_prop" + ts);
              python_block->mutable_python_param()->set_module("block_prop");
              python_block->mutable_python_param()->set_layer("block_prop");
              python_block->add_bottom("cur_action_score" + ts);
              python_block->add_top("cur_action_score_blocked" + ts);
          }
       
          {
              LayerParameter* eliwiseproduct_layer = net_param->add_layer();
              eliwiseproduct_layer->CopyFrom(eliwise_weight);
              eliwiseproduct_layer->set_name("weight_action_prev1" + ts);
              caffe::ParamSpec* weight = eliwiseproduct_layer->add_param();
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              eliwiseproduct_layer->add_bottom("cur_action_score_blocked" + ts);
              eliwiseproduct_layer->add_top("cur_action_score_weighted1" + ts);
          }
          {
              LayerParameter* eliwiseproduct_layer = net_param->add_layer();
              eliwiseproduct_layer->CopyFrom(eliwise_weight);
              eliwiseproduct_layer->set_name("weight_action_prev2" + ts);
              caffe::ParamSpec* weight = eliwiseproduct_layer->add_param();
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              eliwiseproduct_layer->add_bottom("cur_action_score_blocked" + ts);
              eliwiseproduct_layer->add_top("cur_action_score_weighted2" + ts);
          }
      }
      // Add layer to normalize prediction
      {
          LayerParameter* softmax_layer_normalize_action = net_param->add_layer();
          softmax_layer_normalize_action->set_type("Softmax");
          softmax_layer_normalize_action->set_name("Normalize_Action_Pred" + ts);
          softmax_layer_normalize_action->add_bottom("cur_action_score" + ts);
          softmax_layer_normalize_action->add_top("cur_action_score_normalized" + ts);
          softmax_layer_normalize_action->mutable_softmax_param()->set_axis(2);
      }
      // add reshape layer to clean those dummy data
      {
		  LayerParameter* python_filter = net_param->add_layer();
		  python_filter->set_name("python_filter" + ts);
		  python_filter->set_type("Python");
		  python_filter->mutable_python_param()->set_module("filter_action");
		  python_filter->mutable_python_param()->set_layer("filter_action");
		  python_filter->add_bottom("cur_action_score_normalized" + ts);
          python_filter->add_bottom("label_action");
		  python_filter->add_top("cur_action_score_normalized_reshaped" + ts);
	  }
      }
      else{
          {
			  LayerParameter* python_scene = net_param->add_layer();
			  python_scene->set_name("python_scene" + ts);
			  python_scene->set_type("Python");
			  python_scene->mutable_python_param()->set_module("sum_scene");
			  python_scene->mutable_python_param()->set_layer("sum_scene");
			  python_scene->add_bottom("Message_1"+ts);
              python_scene->add_bottom("label_action");
			  python_scene->add_top("scene_score" + ts);
		  }
          {
			  LayerParameter* python_action = net_param->add_layer();
			  python_action->set_name("python_action" + ts);
			  python_action->set_type("Python");
			  python_action->mutable_python_param()->set_module("sum_action");
			  python_action->mutable_python_param()->set_layer("sum_action");
			  python_action->add_bottom("Message_2"+ts);
              python_action->add_bottom("label_action");
			  python_action->add_top("cur_action_score" + ts);
		  }
      }
      // check scene differentials by python layer
      /*{
		  LayerParameter* python_checkdiff = net_param->add_layer();
		  python_checkdiff->set_name("python_checkdiff_scene" + ts);
		  python_checkdiff->set_type("Python");
		  python_checkdiff->mutable_python_param()->set_module("check_diff");
		  python_checkdiff->mutable_python_param()->set_layer("check_diff");
          python_checkdiff->add_bottom("scene_score_forcheck" + ts);
          python_checkdiff->add_top("scene_score"+ts);
	  }*/
      
      /*
      // neuron layers:
      {
          {
		    LayerParameter* h_neuron_param = net_param->add_layer();
		    h_neuron_param->CopyFrom(tanh_param);
		    h_neuron_param->set_name("o_neuron1_" + ts);
		    h_neuron_param->add_bottom("scene_score_neuron_input_" + ts);
		    h_neuron_param->add_top("scene_score" + ts);
		  }
          {
		    LayerParameter* h_neuron_param = net_param->add_layer();
		    h_neuron_param->CopyFrom(tanh_param);
		    h_neuron_param->set_name("o_neuron2_" + ts);
		    h_neuron_param->add_bottom("cur_action_score_neuron_input_" + ts);
		    h_neuron_param->add_top("cur_action_score" + ts);
		  }
      }*/
      if (this->K_>0){
      // Add latent boost layer: Temporal Frame action node prediction, leaf nodes
      {
          LayerParameter* Latent_Boost_Temporal_leaf = net_param->add_layer();
          Latent_Boost_Temporal_leaf->CopyFrom(temporal1);
          Latent_Boost_Temporal_leaf->set_name("LB_temporal_leaf" + ts);
          caffe::ParamSpec* weight = Latent_Boost_Temporal_leaf->add_param();
          caffe::ParamSpec* weight_bias = Latent_Boost_Temporal_leaf->add_param();
          if (separate_initial_weight){
             if (t == 1){
                 weight->set_name("W_ho_3" + ts);
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
             else{
                 weight->set_name("W_ho_3");
                 weight->set_lr_mult(w_lr_mult);
                 weight->set_decay_mult(w_decay_mult);
                 weight_bias->set_lr_mult(b_lr_mult);
                 weight_bias->set_decay_mult(b_decay_mult);
             }
          }
          else{
              weight->set_name("W_ho_3");
              weight->set_lr_mult(w_lr_mult);
              weight->set_decay_mult(w_decay_mult);
              weight_bias->set_lr_mult(b_lr_mult);
              weight_bias->set_decay_mult(b_decay_mult);
          }
          Latent_Boost_Temporal_leaf->add_bottom("Message_3" + ts);
          Latent_Boost_Temporal_leaf->add_top("tem_action_score_leaf" + ts);
          Latent_Boost_Temporal_leaf->mutable_inner_product_param()->set_axis(3); //---------------------- set axis
      }
      }
      if (context){
      LayerParameter* context_param = net_param->add_layer();
      context_param->CopyFrom(sum_param);
      //context_param->mutable_eltwise_param()->set_coeff_blob(true);
      context_param->set_name("sum_context_" + ts);
      context_param->add_bottom("scene_score" + ts);
      context_param->add_bottom("fc8_context");
      context_param->add_top("contexted_scene_score" + ts);
      }

      if (this->K_>1){
          
		  // Add latent boost layer: Temporal Frame action node prediction, non leaf nodes
		  {
		      LayerParameter* Latent_Boost_Temporal_mid = net_param->add_layer();
		      Latent_Boost_Temporal_mid->CopyFrom(temporal1);
		      Latent_Boost_Temporal_mid->set_name("LB_temporal_mid" + ts);
              caffe::ParamSpec* weight = Latent_Boost_Temporal_mid->add_param();
              caffe::ParamSpec* weight_bias = Latent_Boost_Temporal_mid->add_param();
              if (separate_initial_weight){
                  if (t == 1){
                      weight->set_name("W_ho_4" + ts);
                      weight->set_lr_mult(w_lr_mult);
                      weight->set_decay_mult(w_decay_mult);
                      weight_bias->set_lr_mult(b_lr_mult);
                      weight_bias->set_decay_mult(b_decay_mult);
                  }
                  else{
                      weight->set_name("W_ho_4");
                      weight->set_lr_mult(w_lr_mult);
                      weight->set_decay_mult(w_decay_mult);
                      weight_bias->set_lr_mult(b_lr_mult);
                      weight_bias->set_decay_mult(b_decay_mult);
                  }
              }
              else{
                  weight->set_name("W_ho_4");
                  weight->set_lr_mult(w_lr_mult);
                  weight->set_decay_mult(w_decay_mult);
                  weight_bias->set_lr_mult(b_lr_mult);
                  weight_bias->set_decay_mult(b_decay_mult);
              }
		      Latent_Boost_Temporal_mid->add_bottom("Message_4" + ts);
		      Latent_Boost_Temporal_mid->add_top("tem_action_score_mid" + ts);
		      Latent_Boost_Temporal_mid->mutable_inner_product_param()->set_axis(3); //---------------------- set axis
		  }
          // Add layer to reshape message to desired shape
          {
		      LayerParameter* python_reshape3 = net_param->add_layer();
		      python_reshape3->set_name("python_reshape3" + ts);
		      python_reshape3->set_type("Python");
		      python_reshape3->mutable_python_param()->set_module("Message_Reshape3");
		      python_reshape3->mutable_python_param()->set_layer("Message_Reshape3");
		      python_reshape3->add_bottom("tem_action_score_mid"+ts);
		      python_reshape3->add_top("tem_action_score_mid_reshaped" + ts);
          }
      }
      // Add layer to reshape message to desired shape
      {
	      LayerParameter* python_reshape1 = net_param->add_layer();
	      python_reshape1->set_name("python_reshape1" + ts);
	      python_reshape1->set_type("Python");
	      python_reshape1->mutable_python_param()->set_module("Message_Reshape1");
	      python_reshape1->mutable_python_param()->set_layer("Message_Reshape1");
	      python_reshape1->add_bottom("cur_action_score"+ts);
	      python_reshape1->add_top("cur_action_score_reshaped" + ts);
      }
      if (this->K_>0){
      // Add layer to reshape message to desired shape
      {
	      LayerParameter* python_reshape2 = net_param->add_layer();
	      python_reshape2->set_name("python_reshape2" + ts);
	      python_reshape2->set_type("Python");
	      python_reshape2->mutable_python_param()->set_module("Message_Reshape2");
	      python_reshape2->mutable_python_param()->set_layer("Message_Reshape2");
	      python_reshape2->add_bottom("tem_action_score_leaf"+ts);
	      python_reshape2->add_top("tem_action_score_leaf_reshaped" + ts);
      }
      }

      {
		  LayerParameter* python_checkdiff = net_param->add_layer();
		  python_checkdiff->set_name("python_checkdiff" + ts);
		  python_checkdiff->set_type("Python");
		  python_checkdiff->mutable_python_param()->set_module("check_diff");
		  python_checkdiff->mutable_python_param()->set_layer("check_diff");
          python_checkdiff->add_bottom("cur_action_score_reshaped"+ts);
          python_checkdiff->add_top("cur_action_score_reshaped_checked"+ts);
	  }
      
      // Add  every time step's output to concatenation layer
      if (t > timecontrol){
		  if (context){
             scene_output_concat_layer.add_bottom("contexted_scene_score"+ts); 
          }
          else{
             scene_output_concat_layer.add_bottom("scene_score"+ts); 
             scene_output_python.add_bottom("scene_score"+ts); 
          }

		  //action_output_concat_layer.add_bottom("cur_action_score_reshaped_checked" + ts);
          action_output_python.add_bottom("cur_action_score_reshaped_checked" + ts);
      }
      if (this->K_>0){
          action_output_concat_layer.add_bottom("tem_action_score_leaf_reshaped" + ts);
      }
      if (this->K_>1){
          action_output_concat_layer.add_bottom("tem_action_score_mid_reshaped" + ts); 
      }     
    }

    //net_param->add_layer()->CopyFrom(scene_output_concat_layer);  //  T_*N_*5
    net_param->add_layer()->CopyFrom(scene_output_python);
    //net_param->add_layer()->CopyFrom(action_output_concat_layer);  // T_*N_*[(2k+1)*num_people]*40 
    net_param->add_layer()->CopyFrom(action_output_python);

    {
	  LayerParameter* python_checkdiff_o = net_param->add_layer();
	  python_checkdiff_o->set_name("python_checkdiff_o");
	  python_checkdiff_o->set_type("Python");
	  python_checkdiff_o->mutable_python_param()->set_module("check_diff");
	  python_checkdiff_o->mutable_python_param()->set_layer("check_diff");
      python_checkdiff_o->add_bottom("o_action_forcheck");
      python_checkdiff_o->add_bottom("o_action_forcheck");
      if (ifforprint){
          python_checkdiff_o->add_top("action_pred");
      }else{
          python_checkdiff_o->add_top("o_action");
      }
      //python_checkdiff_o->add_bottom("o_scene_forcheck");
      //python_checkdiff_o->add_bottom("o_scene_forcheck");
      //python_checkdiff_o->add_top("o_scene");
    }

    /*{
	      LayerParameter* python_silence = net_param->add_layer();
	      python_silence->set_name("python_silence");
	      python_silence->set_type("Python");
	      python_silence->mutable_python_param()->set_module("Silence");
	      python_silence->mutable_python_param()->set_layer("Silence_Layer");
	      python_silence->add_bottom("cur_action_score1");
 	      python_silence->add_bottom("scene_score1");
	      //python_silence->add_top("tem_action_score_leaf_reshaped" + ts);
    }*/
}

INSTANTIATE_CLASS(BP_RNNLayer);
REGISTER_LAYER_CLASS(BP_RNN);

}  // namespace caffe
