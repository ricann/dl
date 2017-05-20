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
    int cnt = blob.count();
    float *data = blob.mutable_cpu_data();
    float *diff = blob.mutable_cpu_diff();
    for(int i=0; i<cnt; i++) {
        data[i] = i;
        diff[i] = cnt -1 -i;
    }
    blob.Update();
    cout << "blob.count() = " << cnt << endl;

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

    cout << "ASUM = " << blob.asum_data() << endl;
    cout << "SUMSQ = " << blob.sumsq_data() << endl;

    return 0;
}
