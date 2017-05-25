#include <caffe/net.hpp>
#include "common.hpp"

using namespace caffe;

int main()
{
    string proto("deploy.prototxt");
    Net<float> nn(proto, caffe::TEST);
    vector<string> bn = nn.blob_names();

    for(int i=0; i<bn.size(); i++) {
        cout << "Blob #" << i << " : " << bn[i] << endl;
    }

    return 0;
}
