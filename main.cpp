#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace std;

using namespace tiny_dnn;
using namespace tiny_dnn::layers;
using namespace tiny_dnn::activation;

int main()
{
	adam opt;
	network<sequential> net;
	int layer_in = 6;
	int layer_mid1 = 20;
	int layer_mid2 = 20;
	int layer_out = 11;

	net << fc(layer_in,layer_mid1) << tanh_layer() << fc(layer_mid1,layer_mid2)
	<< tanh_layer()	<< fc(layer_mid2,layer_out) << softmax(); 
	cout<<"make neural net"<<endl;

	int data_size = 10;
	std::vector<vec_t> input_data;
	std::vector<label_t> desired_out(data_size);
	cout<<"make data var"<<endl;

	for(int i = 0;i<data_size;i++)
	{
		int label_id;
		cin >> label_id;
		desired_out[i] = label_id+1;
	
		vec_t x(layer_out);
		for(int j =0;j<layer_out;j++)
			cin>>x[i];
		input_data.push_back(x);
		cout<<"c"<<endl;
	}
	cout<<"make data"<<endl;

	int batch_size = 1;
	int epochs = 1;
	net.train<mse>(opt,input_data,desired_out,batch_size,epochs);
	cout<<"end train"<<endl;

	for(int i=0;i<10;i++)
	{
		cout<<net.predict_label(input_data[i])<<endl;
	}

}
