#ifndef __TEST_COMMON_H__
#define __TEST_COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <assert.h>

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <thread>
#include <mutex>
#include <memory>
#include <queue>

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

using std::string;
using std::cout;
using std::endl;
using std::fstream;
using std::vector;
using std::thread;
using std::mutex;
using std::shared_ptr;

using std::hex;
using std::left;
using std::setw;
using std::setfill;


#endif /* __TEST_COMMON_H__ */

