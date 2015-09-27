#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EliwiseProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //const int num_output = this->layer_param_.eliwise_product_param().num_output();
  bias_term_ = this->layer_param_.eliwise_product_param().bias_term();
  //N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.eliwise_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N eliwise products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> weight_shape(1);

    int strategy = 2;

    if (strategy == 1){
        weight_shape[0] = K_;
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.eliwise_product_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
    }
    else{
        weight_shape[0] = 1;
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.eliwise_product_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
    }

    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.eliwise_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void EliwiseProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.eliwise_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with eliwise product parameters.";
  // The first "axis" dimensions are independent eliwise products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = K_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(1, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void EliwiseProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int strategy = 2;
  if (strategy == 1){
	  //const Dtype* weight = this->blobs_[0]->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
	  //const int count = top[0]->count();
	  for (int i = 0; i < M_; ++i){
	      caffe_mul(K_, bottom[0]->cpu_data() + i*K_, this->blobs_[0]->cpu_data(), top_data + i*K_);
	  }
  }
  else{
	  //const Dtype* weight = this->blobs_[0]->cpu_data();
	  Dtype* top_data = top[0]->mutable_cpu_data();
	  //const int count = top[0]->count();
	  for (int i = 0; i < M_; ++i){
	      caffe_cpu_scale(K_, *this->blobs_[0]->cpu_data(), bottom[0]->cpu_data() + i*K_, top_data + i*K_);
	  }
  }
}

template <typename Dtype>
void EliwiseProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  int strategy = 2;
  if (strategy == 1){
	  if (this->param_propagate_down_[0]) {
	    // save gradient for weight
	    //Dtype* weight_diff = new Dtype[K_];
	    Dtype* weight_diff_per = new Dtype[K_]; 
	    // Gradient with respect to weight
	    //Dtype* weight = this->blobs_[0]->mutable_cpu_diff();
	    for (int i = 0; i < M_; ++i){
		caffe_mul(K_, top[0]->cpu_diff() + i*K_, bottom[0]->cpu_data() + i*K_, weight_diff_per);
		caffe_add(K_, this->blobs_[0]->mutable_cpu_diff(), weight_diff_per, this->blobs_[0]->mutable_cpu_diff());      
	    }
	    //std::cout << weight[0] << " " << weight[1] << " " << weight[2] << std::endl;
	  }

	  if (propagate_down[0]) {
	    //const Dtype* top_diff = top[0]->cpu_diff();
	    // Gradient with respect to bottom data
	    for (int i = 0; i < M_; ++i){
		caffe_mul(K_, top[0]->cpu_diff() + i*K_, this->blobs_[0]->cpu_data(), bottom[0]->mutable_cpu_diff()  + i*K_);
	    }
	  }
  }
  else{
          if (this->param_propagate_down_[0]) {
	    // save gradient for weight
	    //Dtype* weight_diff = new Dtype[K_];
	    Dtype* weight_diff_per = new Dtype[K_]; 
	    // Gradient with respect to weight
	    //Dtype* weight = this->blobs_[0]->mutable_cpu_diff();
	    for (int i = 0; i < M_; ++i){
		caffe_mul(K_, top[0]->cpu_diff() + i*K_, bottom[0]->cpu_data() + i*K_, weight_diff_per);
                Dtype diff_per = caffe_cpu_asum(K_, weight_diff_per);
		caffe_add(1, this->blobs_[0]->mutable_cpu_diff(), &diff_per, this->blobs_[0]->mutable_cpu_diff());      
	    }
	  }

	  if (propagate_down[0]) {
	    //const Dtype* top_diff = top[0]->cpu_diff();
	    // Gradient with respect to bottom data
	    for (int i = 0; i < M_; ++i){
		caffe_cpu_scale(K_, *this->blobs_[0]->cpu_data(), top[0]->cpu_diff() + i*K_, bottom[0]->mutable_cpu_diff()  + i*K_);
	    }
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EliwiseProductLayer);
#endif

INSTANTIATE_CLASS(EliwiseProductLayer);
REGISTER_LAYER_CLASS(EliwiseProduct);

}  // namespace caffe
