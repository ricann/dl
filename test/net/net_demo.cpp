#include <caffe/net.hpp>
#include "common.hpp"

using namespace caffe;

int main()
{
    FLAGS_log_dir = "./log";
    google::InitGoogleLogging("net_demo");

    string proto("deploy.prototxt");
    Net<float> nn(proto, caffe::TEST);

    cout << endl;
    cout << "blob list ..." << endl;
    vector<string> bn = nn.blob_names();
    for(int i=0; i<bn.size(); i++) {
        cout << "Blob #" << i << " : " << bn[i] << endl;
    }

    cout << endl;
    cout << "layer list ..." << endl;
    vector<string> ln = nn.layer_names();
    for(int i=0; i<ln.size(); i++) {
        cout << "layer #" << i << " : " << ln[i] << endl;
    }

    google::ShutdownGoogleLogging();
    return 0;
}
