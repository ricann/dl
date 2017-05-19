#include <vector>
#include <iostream>
#include <caffe/blob.hpp>

using namespace caffe;
using namespace std;

int main()
{
    Blob<float> a;
    cout << "Size : " << a.shape_string() << endl;
    a.Reshape(1, 2, 3, 4);
    cout << "Size : " << a.shape_string() << endl;

    float *p = a.mutable_cpu_data();
    for(int i=0; i<a.count(); i++)
        p[i] = i;
    cout << "a.count() = " << a.count() << endl;

    return 0;
}
