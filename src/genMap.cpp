#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "genmap.h"

namespace plt = matplotlibcpp;

int main() {
    plt::plot({1,3,2,4});
    plt::show();
    return 0;
}