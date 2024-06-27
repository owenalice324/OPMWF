# TLS input path
    path_dir=~/WHU-TLS/facade/scans/;#input directory
    out_dir=~/WHU-TLS/facade/resultfinal/;#output directory
    pairs_dir=~/WHU-TLS/facade/pairs.txt;#output directory
#parameters
	block_size=5;# block size for hierarchical regisration (default: 5)
    downsample_size=0.1;# downsample size of input TLS scans for coarse registration (default: 0.1 m)
	downsample_size_icp=0.1;# distance of searching corresponding point in icp (default: the same as ${downsample_size})
	lum_iter=3;# number of iteration for lum optimization (default: 3)
	t_MCS=10;# threshold of MCS (default: 10)
	number_of_threads=16;# number of threads you use (default: 16)
	visualize_each_block=0;# visualize each scan-block (default: 0)

#run
./build/OPMWF ${path_dir} ${out_dir} \
 ${block_size}\
 ${downsample_size}\
 ${downsample_size_icp}\
 ${lum_iter} ${t_MCS} \
 ${number_of_threads}\
 ${visualize_each_block}\
 ${pairs_dir}\
