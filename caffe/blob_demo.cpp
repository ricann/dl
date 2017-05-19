#include <vector>
#include <iostream>
#include <caffe/blob.hpp>

using namespace caffe;
using namespace std;

int main()
{
    Blob<float> blob;
    cout << "Size : " << blob.shape_string() << endl;
    blob.Reshape(1, 2, 3, 4);
    cout << "Size : " << blob.shape_string() << endl;

    // set data and print them
    float *data = blob.mutable_cpu_data();
    for(int i=0; i<blob.count(); i++)
        data[i] = i;
    cout << "blob.count() = " << blob.count() << endl;

    for(int n=0; n<blob.shape(0); n++) {
        for(int c=0; c<blob.shape(1); c++) {
            for(int h=0; h<blob.shape(2); h++) {
                for(int w=0; w<blob.shape(3); w++) {
                    cout << "blob"
                        << "[" << n << "]"
                        << "[" << c << "]"
                        << "[" << h << "]"
                        << "[" << w << "]"
                        << " = "
                        << blob.data_at(n, c, h, w) << endl;
                }
            }
        }
    }

    return 0;
}
