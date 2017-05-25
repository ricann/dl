#include <vector>
#include <iostream>
#include <caffe/blob.hpp>
#include <caffe/util/io.hpp>

using namespace caffe;
using namespace std;

void PrintBlob(Blob<float> &blob);

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

    // print blob data, asum, sumsq
    PrintBlob(blob);

    BlobProto bp1;
    blob.ToProto(&bp1, true);
    WriteProtoToBinaryFile(bp1, "./blob.txt");

    BlobProto bp2;
    ReadProtoFromBinaryFile("./blob.txt", &bp2);
    Blob<float> blob2;
    blob2.FromProto(bp2, true);

    PrintBlob(blob2);

    return 0;
}

void PrintBlob(Blob<float> &blob)
{
    cout << endl;
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
    blob.scale_data(0.1);
    ReshapeLike(source);
    cout << endl;
}
