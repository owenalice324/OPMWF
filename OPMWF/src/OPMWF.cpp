#include "../include/OPMWF.h"

BASE::OPMWF::OPMWF():
	voxel_size(0.2),
	voxel_size_icp(0.1),
	use_cyclic_constraints(1),
	use_LUM(1),
	translation_accuracy(0.5),
	rotation_accuracy(0.087),
	approx_overlap(0.3),
	part_num(5),
	check_blcok(false),
	bigloop(0),
	all_count(0),
	use_pairs(true),
	max_consensus_set(10),
	num_threads(1)
{

}
BASE::OPMWF::~OPMWF()
{
}



void BASE::OPMWF::buildgraph()
{
	
	pairs.clear();
	filename.str("");
	nr_scans = 0;
	

	if (use_pairs)
	{
		//readPairs(pair_path);
		//readPairs();
		
		//readPairsFromFile(); 
		
		
		nr_scans = files.size();
		pairs.clear();
		cout << "MakeCompletepair " << endl;
		for (int i = 1; i < nr_scans+1; i++)
		{
			for (int j = i + 1; j < nr_scans+1; j++)
			{

				pairs.push_back(std::make_pair(j, i));
				cout << "push pair(" <<j<<","<<i<<")" << endl;
				

			}
		}
	}
	else
	{
		nr_scans = files.size();
		pairs.clear();
		cout << "MakeCompletepair " << endl;
		for (int i = 1; i < nr_scans+1; i++)
		{
			for (int j = i + 1; j < nr_scans+1; j++)
			{

				pairs.push_back(std::make_pair(i, j));
				cout << "push pair(" <<i<<","<<j<<")" << endl;
				

			}
		}
	}
}



int BASE::OPMWF::readPLYfiles()
{
	if (!boost::filesystem::exists(PLYpath))
	{
		cerr << "...path does not exists!\n";
		return 0;
	}
	if (!boost::filesystem::is_directory(PLYpath))
	{
		cerr << "...path is not a directory!\n";
		return 0;
	}
	for (boost::filesystem::directory_entry& x : boost::filesystem::directory_iterator(PLYpath))
	{
		files.push_back(x.path());
	}
	std::sort(files.begin(),files.end());
	return 1;
}

//Voxelgriddownsample
int BASE::OPMWF::sampleLeafsized( pcl::PointCloud<PointT>::Ptr& cloud_in, 
	pcl::PointCloud<PointT>& cloud_out, 
	float downsample_size)
{

	pcl::PointCloud <PointT> cloud_sub;
	cloud_out.clear();
	float leafsize = downsample_size * (std::pow(static_cast <int64_t> (std::numeric_limits <int32_t>::max()) - 1, 1. / 3.) - 1);

	pcl::octree::OctreePointCloud <PointT> oct(leafsize); // new octree structure
	oct.setInputCloud(cloud_in);
	oct.defineBoundingBox();
	oct.addPointsFromInputCloud();

	pcl::VoxelGrid <PointT> vg; // new voxel grid filter
	vg.setLeafSize(downsample_size, downsample_size, downsample_size);
	vg.setInputCloud(cloud_in);

	size_t num_leaf = oct.getLeafCount();

	pcl::octree::OctreePointCloud <PointT>::LeafNodeIterator it = oct.leaf_depth_begin(), it_e = oct.leaf_depth_end();
	for (size_t i = 0; i < num_leaf; ++i, ++it)
	{
		pcl::IndicesPtr ids(new std::vector <int>); // extract octree leaf points
		pcl::octree::OctreePointCloud <PointT>::LeafNode* node = (pcl::octree::OctreePointCloud <PointT>::LeafNode*) * it;
		node->getContainerPtr()->getPointIndices(*ids);

		vg.setIndices(ids); // set cloud indices
		vg.filter(cloud_sub); // filter cloud

		cloud_out += cloud_sub; // add filter result
	}

	return (static_cast <int> (cloud_out.size())); // return number of points in sampled cloud
}

void BASE::OPMWF::readPointCloud(const boost::filesystem::path& filename,
	pcl::PointCloud<PointT>::Ptr cloud
)
{
	if (!filename.extension().string().compare(".ply"))
	{
		pcl::io::loadPLYFile(filename.string(), *cloud);
		return;
	}
	if (!filename.extension().string().compare(".pcd"))
	{
		pcl::io::loadPCDFile(filename.string(), *cloud);
		return;
	}
}

double computeOtsuThreshold(const std::vector<double>& data) {
    if (data.empty()) return 0.0;

    // Histogram settings
    const size_t num_bins = 256;
    auto min_max = std::minmax_element(data.begin(), data.end());
    double min = *min_max.first;
    double max = *min_max.second;
    std::vector<int> histogram(num_bins, 0);

    // Create histogram
    double bin_width = (max - min) / num_bins;
    for (double d : data) {
        int bin = std::min(num_bins - 1, static_cast<size_t>((d - min) / bin_width));
        histogram[bin]++;
    }

    // Otsu's method
    int total = data.size();
    double sum = 0;
    for (size_t i = 0; i < num_bins; ++i) {
        sum += i * histogram[i];
    }

    double sumB = 0, wB = 0, wF = 0, mB, mF, max_var = 0, threshold = 0;
    for (size_t i = 0; i < num_bins; ++i) {
        wB += histogram[i];
        if (wB == 0) continue;
        wF = total - wB;
        if (wF == 0) break;

        sumB += i * histogram[i];
        mB = sumB / wB;
        mF = (sum - sumB) / wF;

        double varBetween = wB * wF * (mB - mF) * (mB - mF);
        if (varBetween > max_var) {
            max_var = varBetween;
            threshold = i;
        }
    }

    return min + threshold * bin_width;
}


void BASE::OPMWF::caculatefpfh(std::vector<pcl::PointCloud<PointT>::Ptr>& keypoint_clouds, 
	std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& keypoint_clouds_feature,
	std::vector<pcl::PointCloud<PointT>::Ptr>& clouds
)
{
	cout << "keypoint_size== " << keypoint_clouds.size() << ":" << endl;

	std::vector<Eigen::Vector4f> centroids(keypoint_clouds.size());
    	Eigen::MatrixXd distance_matrix(keypoint_clouds.size(), keypoint_clouds.size());
#pragma omp parallel for num_threads(num_threads) 
	for (int i = 0; i < nr_scans; i++)
	{
		cout << "Processing " <<i << ":";

		// load point cloud
		cout << "..loading point cloud..\n";
		pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
		readPointCloud(files[i], cloud);
		// color(cloud);
		clouds[i] = cloud;
		//cout << "ok!" << endl;

		// voxelgrid  downsample 
		cout << "..apply voxel grid filter..\n";
		pcl::PointCloud <PointT>::Ptr voxel_cloud(new pcl::PointCloud <PointT>);
		pcl::PointCloud<PointT>::Ptr voxel_cloud_icp(new pcl::PointCloud<PointT>);
		sampleLeafsized(clouds[i], *voxel_cloud, voxel_size);
		sampleLeafsized(clouds[i], *voxel_cloud_icp, voxel_size_icp);
		clouds[i] = voxel_cloud_icp;
		//cout << "ok!" << endl;

		//iss
		pcl::PointCloud<PointT>::Ptr issS(new pcl::PointCloud<PointT>);
		pcl::PointIndicesPtr issIdxS(new pcl::PointIndices);
		std::cout << "extracting ISS keypoints..." << voxel_size << std::endl;
		GrorPre::issKeyPointExtration(voxel_cloud, issS, issIdxS, voxel_size);
		std::cout << "size of issS = " << issS->size() << std::endl;
		issS->is_dense = false;
		keypoint_clouds[i] = issS;
		std::string str = std::to_string(i);
		pcl::io::savePCDFile(output_dir + "/keypoint/keypoint"+str+".pcd", *keypoint_clouds[i],true);
		//fpfh
		std::cout << "computing fpfh..." << std::endl << std::endl;;

		pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhS(new pcl::PointCloud<pcl::FPFHSignature33>());
		GrorPre::fpfhComputation(voxel_cloud, voxel_size, issIdxS, fpfhS);
		keypoint_clouds_feature[i] = fpfhS;
		cout << "ok!\n";

		pcl::compute3DCentroid(*keypoint_clouds[i], centroids[i]);
		std::cout << "computing Centroid..." << std::endl ;
		

	}
#pragma omp critical 
	
	    // Calculate pairwise distances between centroids
	    for (int i = 0; i < centroids.size(); ++i) 
	    {
		for (int j = 0; j < centroids.size(); ++j) 
		{
		    if (i == j) {
		        distance_matrix(i, j) = 0;
		    } else {
		        distance_matrix(i, j) = (centroids[i].head<3>() - centroids[j].head<3>()).norm();
		    }
		}
	    }

	    // Optionally print or store the distance matrix
	    std::cout << "Distance Matrix: \n" << distance_matrix << std::endl;
	    
	    // Average
	    double sum = 0;
	    double avg = 0;
	    int count = 0;
	    for (int i = 0; i < distance_matrix .rows(); ++i) {
		for (int j = i + 1; j < distance_matrix .cols(); ++j) {  // Skip diagonal and duplicate entries
		    sum += distance_matrix (i, j);
		    ++count;
		}
	    }
	    avg= count > 0 ? sum / count : 0;
	    std::cout << "Average threshold=:" << avg << std::endl;
	    
	    
	     // Flatten the matrix to a vector excluding the diagonal
	    std::vector<double> distances;
	    for (int i = 0; i < distance_matrix.rows(); ++i) {
		for (int j = i + 1; j < distance_matrix.cols(); ++j) {
		    distances.push_back(distance_matrix(i, j));
		}
	    }

	    // Compute the Otsu's threshold
	    double otsu_threshold = computeOtsuThreshold(distances);
	    std::cout << "Otsu's Threshold: " << otsu_threshold << std::endl;
	    
	    
	    std::vector<std::pair<int, int>> filtered_pairs;
	    for (const auto& pair : pairs) 
	    {
		int i = pair.first-1;
		int j = pair.second-1;
		if (distance_matrix(j, i) <=18) 
		{  // 刪除矩阵中对应值大于或等于7的对
		    filtered_pairs.push_back(pair);
        	}
    	    }
    	    // 将 filtered_pairs 的内容复制回 pairs
	    pairs.clear();           // 首先清空原始的 pairs 向量
	    pairs = filtered_pairs;  // 然后将 filtered_pairs 的内容复制给它
    	    
    	    
    	    for (const auto& pair : pairs) 
    	    {
        	std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
    	    }
	    

}

void BASE::OPMWF::coarseRgistration(std::vector<pcl::PointCloud<PointT>::Ptr>& keypoint_clouds, 
	std::vector<pcl::PointCloud<pcl::FPFHSignature33>::Ptr>& keypoint_clouds_feature,
	std::vector<pcl::PointCloud<PointT>::Ptr>& clouds, 
	std::vector<int>& pairs_best_count,
	std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& candidate_matches
)
{

	for (int i = 0; i < nr_pairs;i++)
	{
		cout << "-";
	}
	cout << "\n";
	if (method == CoarseRegistration::GROR)
	{
#pragma omp parallel for num_threads(num_threads)
		for (int i = 0; i < nr_pairs; i++)
		{
			int n_optimal = 800; //optimal selection number
			const int& src = pairs[i].first-1;
			const int& tgt = pairs[i].second-1;

			cout << "Matching keypoints of " << files[src].stem().string() << " and " << files[tgt].stem().string() << ".." << std::flush;
			cout<<"\n";
			int maxCorr = 5;
			pcl::CorrespondencesPtr corr(new pcl::Correspondences);
			std::vector<int> corrNOS, corrNOT;

			GrorPre::correspondenceSearching(keypoint_clouds_feature[src], keypoint_clouds_feature[tgt], *corr, maxCorr, corrNOS, corrNOT);
			//std::cout << "NO. corr = " << corr->size() << std::endl;

			pcl::registration::GRORInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, float> obor;
			pcl::PointCloud<pcl::PointXYZ>::Ptr pcs(new pcl::PointCloud<pcl::PointXYZ>);

			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_src(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_tgt(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::copyPointCloud<PointT, pcl::PointXYZ>(*keypoint_clouds[src], *temp_src);
			pcl::copyPointCloud<PointT, pcl::PointXYZ>(*keypoint_clouds[tgt], *temp_tgt);
			obor.setInputSource(temp_src);
			obor.setInputTarget(temp_tgt);
			obor.setResolution(voxel_size);
			obor.setOptimalSelectionNumber(n_optimal);
			obor.setNumberOfThreads(1);
			obor.setInputCorrespondences(corr);
			obor.setDelta(voxel_size);
			obor.align(*pcs);
			

			pcl::registration::MatchingCandidate result;
			result.transformation = obor.getFinalTransformation();
			result.fitness_score = obor.getMSAC();
			pairs_best_count[i] = obor.getBestCount();
			
#pragma omp critical 
			{
				candidate_matches[i]=result;
				cout << "*";
				cout.clear();
			}

			
		}
		cout << "\n";
	
	}
	

}

void BASE::OPMWF::globalCoarseRegistration(
	std::set<int>& rejected_pairs_,
	std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& candidate_matches,
	std::vector<int> &pairs_best_count,
	int nr_scans
)
{
	
	if (nr_scans >= 3  )
	{
		rejected_pairs_.clear();
		std::cout<<"--> loop-based coarse regisration. \n";

		//Loop-coarse registration method. 
		ScanGraphInference sgi;
		sgi.setScanPairs(pairs);
		sgi.setMatchingCandidates(candidate_matches);
		sgi.setMCS(pairs_best_count);
		sgi.setRotationAccuracy(rotation_accuracy);
		sgi.setTranslationAccuracy(translation_accuracy);
		rejected_pairs_ = sgi.inference(3,10);

		std::cout<<"IES size: "<<rejected_pairs_.size()<<"\n";
		
	}
}

void BASE::OPMWF::globalFineRegistration(std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches,
	std::vector <pcl::PointCloud <PointT>::Ptr> &clouds,
	std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
	int &num_of_subtrees
)
{
	if (nr_scans >= 3)
	{
		//temporary varible
		std::vector<std::vector<int>> LUM_indices_map_;
		std::vector<std::vector<int>> LUM_indices_map_inv_;
		std::vector<std::vector<size_t>> edges;
		std::vector<int> root;
		std::vector<std::pair<int, int>> after_check_graph_pairs;
		std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> after_check_matches;
		std::vector<std::vector<pcl::Indices>> nodes_seqs;

		mstCalculation(matches, LUM_indices_map_, LUM_indices_map_inv_, edges, root, after_check_graph_pairs, after_check_matches,num_of_subtrees);
		pairwiseICP(edges, after_check_graph_pairs, after_check_matches, num_of_subtrees, clouds);
		concatenateFineRegistration(edges, root, after_check_graph_pairs, after_check_matches, num_of_subtrees, poses, nodes_seqs);
		LUMoptimization(poses, edges, root, nodes_seqs, LUM_indices_map_, LUM_indices_map_inv_, num_of_subtrees, clouds);
	}
}

void BASE::OPMWF::mstCalculation(std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches,
	std::vector<std::vector<int>>& LUM_indices_map_,
	std::vector<std::vector<int>>& LUM_indices_map_inv_,
	std::vector<std::vector<size_t>> &edges,
	std::vector<int>& root,
	std::vector<std::pair<int, int>>& after_check_graph_pairs,
	std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> &after_check_matches,
	int &num_of_subtrees
)
{
	typedef boost::subgraph<boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::property<boost::vertex_index_t, int>, boost::property<boost::edge_index_t, int, boost::property<boost::edge_weight_t, float>>>> Graph;
	//typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS > Graph;
	Graph G(nr_scans);

	cout << "compute MST\n";
	std::vector<float>  weight;
	for (std::size_t i = 0; i < nr_pairs; i++)
	{
		
		const int& src = pairs[i].first-1;
		const int& tgt = pairs[i].second-1;
		add_edge(src, tgt, G);
		weight.push_back(matches[i].fitness_score);
		after_check_matches.push_back(matches[i]);
		after_check_graph_pairs.push_back(std::make_pair(src, tgt));
	}
	std::cout<<"component calculation...";
	std::vector< int > component(num_vertices(G));


	num_of_subtrees = connected_components(G, &component[0]);
	edges.resize(num_of_subtrees);

	cout<<"subgraphs size: "<<num_of_subtrees;
	 std::vector<std::vector<int>> subgraph_vertices(num_of_subtrees);
    //cout << component.size();
    for (int i = 0; i < component.size(); i++)
    {
        subgraph_vertices[component[i]].push_back(i);
        //cout << component[i] << ": " << i << "\t";
    }

	// to address the vertex&edge descriptor issue (global <-> local)
	// LUM use local descriptor
	 Graph G_r(nr_scans);
	Graph* subgraphs= new Graph[num_of_subtrees];

	for (int i = 0; i < num_of_subtrees; i++)
        {
            Graph &g_s = G_r.create_subgraph();
            for (int j = 0; j < subgraph_vertices[i].size(); j++)
            {
                //cout << subgraph_vertices[i][j] << "\t";
                boost::add_vertex(subgraph_vertices[i][j], g_s);
                //cout <<v;
            }
           //cout << "number of vertex of subgraphs: " << boost::num_vertices(g_t);

            cout << "\n";
            subgraphs[i]=g_s;
        }
	
	for (std::size_t i = 0; i < after_check_graph_pairs.size(); i++)
        {
            int g_s =component[after_check_graph_pairs[i].first];
            boost::property_map<Graph, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, subgraphs[g_s]);
            std::size_t src = subgraphs[g_s].global_to_local(after_check_graph_pairs[i].first);
            std::size_t tgt = subgraphs[g_s].global_to_local(after_check_graph_pairs[i].second);
            boost::graph_traits<Graph>::edge_descriptor e;
            bool inserted;

            boost::tie(e, inserted) = boost::add_edge(src, tgt, subgraphs[g_s]);
            weightmap[e] = weight[i];
        }
	cout << "--> find MST edges: \n";

	LUM_indices_map_.resize(num_of_subtrees);
	LUM_indices_map_inv_.resize(num_of_subtrees);
	for (int i = 0; i < num_of_subtrees ; i++)
	{
		LUM_indices_map_[i].resize(nr_scans, 0);
		LUM_indices_map_inv_[i].resize(nr_scans, 0);
	}
	 std::vector<std::vector<boost::graph_traits<Graph>::vertex_descriptor>> p(num_of_subtrees);
	for(int i=0;i<num_of_subtrees;i++)
        {
			p[i].resize(boost::num_vertices(subgraphs[i]));
			if(boost::num_vertices(subgraphs[i])==1)
				continue;
            boost::prim_minimum_spanning_tree(subgraphs[i],&p[i][0]);
        }

	for (int i = 0; i < num_of_subtrees; i++)
        {
            boost::property_map<Graph, boost::edge_index_t>::type edgemap = boost::get(boost::edge_index, subgraphs[i]);
            cout<<"\tMST: " ;
            for (int j = 0; j < p[i].size(); j++)
            {
				LUM_indices_map_[i][subgraphs[i].local_to_global(j)]=j;
				LUM_indices_map_inv_[i][j]=subgraphs[i].local_to_global(j);
                if(p[i][j]==j)
				{
					cout<<"root: "<<p[i][j]<<"\n";
					root.push_back(subgraphs[i].local_to_global(j));
                    continue;
				}
                edges[i].push_back(edgemap[boost::edge(p[i][j],size_t(j),subgraphs[i]).first]);

                std::cout << subgraphs[i].local_to_global(boost::edge(p[i][j],size_t(j),subgraphs[i]).first) << "\t";
            }
            cout<<endl;
        }
	
	delete[] subgraphs;
	
}

void BASE::OPMWF::pairwiseICP(std::vector<std::vector<size_t>>& edges,
	std::vector<std::pair<int, int>>& after_check_graph_pairs,
	std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches,
	int &num_of_subtrees,
	std::vector <pcl::PointCloud <PointT>::Ptr>& clouds
)
{
	for (int f = 0; f < num_of_subtrees ; f++)
	{
		if (edges[f].size() == 0)
			continue;
			ofstream ofs3(output_dir + "Mst_result.txt");
#pragma omp parallel for num_threads(num_threads)
		for (int i = 0; i < edges[f].size(); i++)
		{

			cout << "pairwise icp process ";
			cout << "edge: " << edges[f][i] << endl;
			
			//normal icp matching or higher level mathcing using saved indices
			if (1)
			{

				pcl::PointCloud<pcl::PointXYZ>::Ptr temp(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr temp2(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::copyPointCloud(*clouds[after_check_graph_pairs[edges[f][i]].first], *temp);
				pcl::copyPointCloud(*clouds[after_check_graph_pairs[edges[f][i]].second], *temp2);

				pcl::IterativeClosestPoint<PointT, PointT> icp;
				if (!temp->empty() && !temp2->empty())
				{
	
					icp.setInputSource(temp);
					icp.setInputTarget(temp2);
					icp.setMaxCorrespondenceDistance(voxel_size_icp);
					icp.setUseReciprocalCorrespondences(true);
					icp.setMaximumIterations(10);
					icp.setEuclideanFitnessEpsilon(10e-3);
					icp.align(*temp, after_check_matches[edges[f][i]].transformation);
					after_check_matches[edges[f][i]].transformation = icp.getFinalTransformation();
					
					ofs3 << after_check_graph_pairs[edges[f][i]].first << "--" << after_check_graph_pairs[edges[f][i]].second << "\n";

					
					cout << after_check_graph_pairs[edges[f][i]].first << "--" << after_check_graph_pairs[edges[f][i]].second << "\n";
					cout << after_check_matches[edges[f][i]].transformation << "\n";
					ofs3 << after_check_matches[edges[f][i]].transformation << "\n";

					
				}
				else {
					cout << "ignore" << i << "because one of them is nullptr!";
				}
				
			}

		}
		cout << "icp complete\n";
		ofs3.close();
	}
}

void BASE::OPMWF::concatenateFineRegistration(std::vector<std::vector<size_t>>& edges,
	std::vector<int>& root,
	std::vector<std::pair<int, int>>& after_check_graph_pairs, 
	std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& after_check_matches, 
	int &num_of_subtrees, 
	std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses, 
	std::vector<std::vector<pcl::Indices>>& nodes_seqs)
{
	for (int f = 0; f < num_of_subtrees ; f++)
	{
		std::stringstream ss;
		ss << f;
		if (edges[f].size() == 0)
		{
			std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_t(nr_scans, Eigen::Matrix4f::Zero());
			poses_t[root[f]] = Eigen::Matrix4f::Identity();
			poses.push_back(poses_t);
			std::vector<pcl::Indices> nodes_seq;
			nodes_seqs.push_back(nodes_seq);
			/*ofstream ofs2(R"(D:\Programming\global_consistency\build\result\concatenate registration)" + filename.str() + "_Mst" + ss.str() + ".txt");

			for (auto& pose : poses_t)
			{
				cout << pose << endl;
				ofs2 << pose << endl;
			}
			ofs2.close();*/
			continue;
		}


		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> Mst_trans(nr_scans - 1);
		std::vector<pcl::Indices> Mst(nr_scans);
		for (int i = 0; i < edges[f].size(); i++)
		{
			Mst[after_check_graph_pairs[edges[f][i]].first].push_back(after_check_graph_pairs[edges[f][i]].second);
			Mst[after_check_graph_pairs[edges[f][i]].second].push_back(after_check_graph_pairs[edges[f][i]].first);
		}
		std::vector<std::pair<int, bool>> map_(nr_scans * nr_scans);
		for (std::size_t i = 0; i < edges[f].size(); i++)
		{
			map_[after_check_graph_pairs[edges[f][i]].first * nr_scans + after_check_graph_pairs[edges[f][i]].second] = std::make_pair(static_cast <int> (edges[f][i]), true);
			map_[after_check_graph_pairs[edges[f][i]].second * nr_scans + after_check_graph_pairs[edges[f][i]].first] = std::make_pair(static_cast <int> (edges[f][i]), false);
		}

		std::vector<pcl::Indices> nodes_seq;
		std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> poses_t(nr_scans, Eigen::Matrix4f::Zero());
		depthfirstsearch(nr_scans, root[f], Mst, map_, nodes_seq);
		cout << "depth search complete\n";
		for (std::vector<int> item : nodes_seq)
		{
			cout << "road :";
			for (int i : item)
			{
				cout << i << "\n";
			}
			cout << "\n";
		}
		combineTransformation(nr_scans, root[f], nodes_seq, map_, after_check_matches, poses_t);
		cout << "root " << root[f] << " is the reference frame :\n";
		//ofstream ofs2(R"(D:\Programming\global_consistency\build\result\concatenate registration)" + filename.str() + "_Mst" + ss.str() + ".txt");

		for (Eigen::Matrix4f& pose : poses_t)
		{
			cout << pose << endl;
			//ofs2 << pose << endl;
		}
		//ofs2.close();
		poses.push_back(poses_t);

		nodes_seqs.push_back(nodes_seq);
	}
}

void BASE::OPMWF::depthfirstsearch(int nr_scans, int root, std::vector<pcl::Indices>& Mst, std::vector<std::pair<int, bool>>& map_, std::vector<pcl::Indices>& nodes_seq)
{
	std::vector<bool> visited(nr_scans, false);

	pcl::Indices path(1, root);
	visited[root] = true;
	//auto leafs =Mst[root];
	std::vector<int> indices = Mst[root];
	pcl::Indices::iterator it = indices.begin();

	for (int i = 0; i < indices.size(); i++)
	{
		int n = indices[i];
		next(n, Mst, map_, nodes_seq, visited, path);
	}
}

void BASE::OPMWF::next(int root, std::vector<pcl::Indices>& Mst, std::vector<std::pair<int, bool>>& map_, std::vector<pcl::Indices>& nodes_seq, std::vector<bool> visited, pcl::Indices path)
{
	visited[root] = true;
	path.push_back(root);
	std::vector<int> indices = Mst[root];
	pcl::Indices::iterator it = indices.begin();
	while (it != indices.end())
	{
		if (visited[*it])
			it = indices.erase(it);
		else
			it++;
	}
	if (indices.size() == 0)
	{
		nodes_seq.push_back(path);
		return;
	}

	for (int i = 0; i < indices.size(); i++)
	{
		int n = indices[i];
		next(n, Mst, map_, nodes_seq, visited, path);
	}
}

void BASE::OPMWF::combineTransformation(int nr_scans, int root, 
	std::vector<pcl::Indices>& nodes_seq, std::vector<std::pair<int, bool>>& map_, 
	std::vector<pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>>& matches, 
	std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>& poses)
{
	Eigen::Matrix4f combin = Eigen::Matrix4f::Identity();

	poses[root] = Eigen::Matrix4f::Identity();
	for (int i = 0; i < nodes_seq.size(); i++)
	{
		combin = Eigen::Matrix4f::Identity();
		for (int j = 0; j < nodes_seq[i].size() - 1; j++)
		{
			std::pair<int,bool> edge = map_[nodes_seq[i][j] * nr_scans + nodes_seq[i][j + 1]];
			
			if (edge.second)
				combin *= inverse(matches[edge.first].transformation);
			else combin *= (matches[edge.first].transformation);
			poses[nodes_seq[i][j + 1]] = combin;
		}
	}
}

Eigen::Matrix4f BASE::OPMWF::inverse(Eigen::Matrix4f& mat)
{
	Eigen::Matrix3f R = mat.block(0, 0, 3, 3);
	Eigen::Vector3f t = mat.block(0, 3, 3, 1);

	Eigen::Matrix4f inversed = Eigen::Matrix4f::Identity();
	inversed.block(0, 0, 3, 3) = R.transpose().block(0, 0, 3, 3);
	inversed.block(0, 3, 3, 1) = (-(R.transpose() * t)).block(0, 0, 3, 1);
	return inversed;
}

void BASE::OPMWF::LUMoptimization(std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>>& poses,
	std::vector<std::vector<size_t>>& edges,
	std::vector<int>& root,
	std::vector<std::vector<pcl::Indices>>& nodes_seqs,
	std::vector<std::vector<int>>& LUM_indices_map_,
	std::vector<std::vector<int>>& LUM_indices_map_inv_,
	int num_of_subtrees,
	std::vector <pcl::PointCloud <PointT>::Ptr>& clouds
)
{
	if (use_LUM)
	{

		std::set<int> visited;
		std::vector< std::vector<pcl::CorrespondencesPtr>> correspondences(edges.size());

		//*********************** apply lum global fine registration ***********************//
		for (int f = 0; f < num_of_subtrees ; f++)
		{
			int q = 0;
			std::stringstream ss;
			ss << f;
			if (edges[f].size() < 2)
			{
				ofstream ofs3(output_dir + filename.str() + "_Mst" + ss.str() + ".txt");
				cout << "...saving results to files" << endl;
				for (int i = 0; i < nr_scans; i++)
				{
					if (poses[f][i].isZero())
					{
						continue;
					}
					ofs3 << "affine transformation: \n";

					ofs3 << poses[f][i];

					ofs3 << "\n";
				}
				ofs3.close();
				continue;
			}

			pcl::registration::LUM<PointT> lum;
			cout << "apply lum global matching..\n";

			lum.addPointCloud(clouds[root[f]]);
			for (int i = 0; i < nr_scans; i++)
			{
				if (i == root[f])
					continue;
				if (poses[f][i].isZero())
				{
					continue;
				}
				Eigen::Transform<float, 3, Eigen::Affine>  aff(poses[f][i]);
				Eigen::Vector6f pose;
				pcl::getTranslationAndEulerAngles(aff, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]);
				lum.addPointCloud(clouds[i], pose);
			}

			int iteration_ = LUM_iterations;
			for (int li = 0; li < iteration_; li++)
			{
				visited.clear();

				cout << "...get correspondences\n";
				for (int i = 0; i < nr_scans; i++)
				{
					if (i == root[f])
						continue;
					if (poses[f][i].isZero())
					{
						continue;
					}
					Eigen::Transform<float, 3, Eigen::Affine>  aff(poses[f][i]);
					Eigen::Vector6f pose;
					pcl::getTranslationAndEulerAngles(aff, pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]);
					lum.setPose(LUM_indices_map_[f][i], pose);
				}
				std::vector<pcl::CorrespondencesPtr> correspondences_t;

				for (int i = 0; i < nodes_seqs[f].size(); i++)
				{


#pragma omp parallel for num_threads(num_threads)
					for (int j = 0; j < nodes_seqs[f][i].size() - 1; j++)
					{
						if (std::find(visited.begin(), visited.end(), nodes_seqs[f][i][j + 1]) == visited.end())
						{
							pcl::registration::CorrespondenceEstimation<PointT, PointT> correspondence_estimate;
							pcl::CorrespondencesPtr temp(new pcl::Correspondences);
							pcl::PointCloud<PointT>::Ptr temp2(new pcl::PointCloud<PointT>());
							pcl::transformPointCloud(*clouds[nodes_seqs[f][i][j + 1]], *temp2, inverse(poses[f][nodes_seqs[f][i][j]]) * poses[f][nodes_seqs[f][i][j + 1]]);
							correspondence_estimate.setInputSource(temp2);
							correspondence_estimate.setInputTarget(clouds[nodes_seqs[f][i][j]]);
							correspondence_estimate.determineReciprocalCorrespondences(*temp, voxel_size_icp);
#pragma omp critical
							{
								visited.insert(nodes_seqs[f][i][j + 1]);
								cout << i << ":" << "correspondences sizes: " << (*temp).size() << "\n";
								lum.setCorrespondences(LUM_indices_map_[f][nodes_seqs[f][i][j + 1]], LUM_indices_map_[f][nodes_seqs[f][i][j]], temp);
							}

						}
					}
				}


				lum.setMaxIterations(1);
				lum.setConvergenceThreshold(0.001);
				cout << "perform lum optimization...\n";
				lum.compute();
				pcl::PointCloud<PointT>::Ptr cloud_out(new pcl::PointCloud<PointT>());
				for (int i = 0; i < lum.getNumVertices(); i++)
				{
					poses[f][LUM_indices_map_inv_[f][i]] = lum.getTransformation(i).matrix();
				}
			}
		
			ofstream ofs3(output_dir + filename.str() + "_Mst" + ss.str() + ".txt");
			cout << "...saving results to files" << endl;
			for (int i = 0; i < lum.getNumVertices(); i++)
			{
				ofs3 << "affine transformation: \n";
				ofs3 << lum.getTransformation(i).matrix();
				poses[f][LUM_indices_map_inv_[f][i]] = lum.getTransformation(i).matrix();
				if (i == (lum.getNumVertices() - 1))
					break;
				ofs3 << "\n";
			}
			ofs3.close();
		}
	}
}

void BASE::OPMWF::solveGlobalPose()
{
	for (int i = Hierarchical_block_poses.size()-1; i >0 ; i--)
	{
		int count = 0;
		for (int j = 0; j < Hierarchical_block_poses[i].size(); j++)
		{
			for (int k = 0; k < Hierarchical_block_poses[i][j].size(); k++)
			{
				for (int m = 0; m < Hierarchical_block_poses[i - 1][count].size(); m++)
				{
					Hierarchical_block_poses[i - 1][count][m] = Hierarchical_block_poses[i][j][k] * Hierarchical_block_poses[i - 1][count][m];
				}
				count++;
			}
		}
	}
	int count = 0;
	for (int i = 0; i < Hierarchical_block_poses[0].size(); i++)
	{
		for (int j = 0; j < Hierarchical_block_poses[0][i].size(); j++)
		{
			global_poses[count]= Hierarchical_block_poses[0][i][j];
			count++;
		}
	}



}

void BASE::OPMWF::eliminateClosestPoints(pcl::PointCloud<PointT>::Ptr& src,
	pcl::PointCloud<PointT>::Ptr& tgt, 
	Eigen::Matrix4f& trans, 
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpfs,
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr& fpft)
{
	pcl::CorrespondencesPtr corr(new pcl::Correspondences);
	pcl::registration::CorrespondenceEstimation<PointT, PointT> es;
	es.setInputSource(src);
	es.setInputTarget(tgt);
	es.determineReciprocalCorrespondences(*corr, voxel_size_icp);

	int before = src->points.size();
	pcl::PointCloud<PointT>::Ptr after_earse(new pcl::PointCloud<PointT>());
	int count = 0;
	for (int i = 0; i < src->points.size(); i++)
	{
		if (i == (*corr)[count].index_query)
		{
			count++;
			continue;
		}
		after_earse->points.push_back(src->points[i]);
	}
	src->clear();
	src = after_earse;
	int after = src->points.size();
	cout << before - after << "points have been removed\n";
	count = 0;
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr after_earse_feature(new pcl::PointCloud<pcl::FPFHSignature33>());
	if (fpfs.get() != nullptr)
	{
		for (int i = 0; i < fpfs->points.size(); i++)
		{
			if (i == (*corr)[count].index_query)
			{
				count++;
				continue;
			}
			after_earse_feature->points.push_back(fpfs->points[i]);
		}
		fpfs->clear();
		fpfs = after_earse_feature;
	}
	
}

// Function to perform DFS on the graph
void DFS(int node, std::map<int, std::set<int>>& graph, std::set<int>& visited) {
    // Stack for DFS
    std::vector<int> stack;
    stack.push_back(node);

    while (!stack.empty()) {
        int current = stack.back();
        stack.pop_back();

        // If not visited, mark and continue DFS
        if (visited.find(current) == visited.end()) {
            visited.insert(current);
            // Push all unvisited neighbors to the stack
            for (int neighbor : graph[current]) {
                if (visited.find(neighbor) == visited.end()) {
                    stack.push_back(neighbor);
                }
            }
        }
    }
}

void BASE::OPMWF::performMultiviewRegistration()
{
	auto begin = std::chrono::steady_clock::now();
	//time count
	std::chrono::duration<double> pretmp;
	std::chrono::duration<double> pretmp1;
	//read PLY files
	if (this->PLYpath.empty())
	{
		std::cerr << "no input paths";
		return;
	}

	readPLYfiles();

	


	/*Loop 1:internal block registration.
	* Loop 2:block-to-block registration.(default: only one "new block", higher level regisration code is not provided)
	* Termination criteria: 1.scans_left == 1, all scans are merged.
	*											  2.scans_left == scans_left, some scans can't merge.
	*/
	

	
	std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> block;
	int loop = 0;
	
	buildgraph();
	nr_pairs = pairs.size();
	cout << "pairs count==" << nr_pairs  << "\n";
	for (const auto& pair : pairs) 
	{
		std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
       }
	
	//keypoint
	std::vector <pcl::PointCloud <PointT>::Ptr> keypoint_clouds;
	keypoint_clouds.resize(nr_scans);
	//fpfh
	std::vector <pcl::PointCloud<pcl::FPFHSignature33>::Ptr> keypoint_clouds_feature;
	keypoint_clouds_feature.resize(nr_scans);
	//input pointcloud
	std::vector <pcl::PointCloud <PointT>::Ptr> clouds;
	clouds.resize(nr_scans);

	//temporary varible for global coarse registration
	std::vector<int> pairs_best_count(nr_pairs);
	std::vector <pcl::registration::MatchingCandidate, Eigen::aligned_allocator<pcl::registration::MatchingCandidate>> candidate_matches(nr_pairs);
	std::set<int> rejected_pairs_;

	//temporary varible for global fine registration
	int num_of_subtrees;
	std::vector<std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>> poses;

	
	
	auto prebegin = std::chrono::steady_clock::now();
	caculatefpfh(keypoint_clouds, keypoint_clouds_feature, clouds);
	auto preend = std::chrono::steady_clock::now();

	pretmp = std::chrono::duration_cast<std::chrono::duration<double>>(preend - prebegin);
	int min1 = pretmp .count()/ 60;
	double sec2 = pretmp .count()- 60 * min1;
	cout << "total cost time:" << min1 << "min" << sec2 << "sec\n";

	
	nr_pairs = pairs.size();
	cout << "pairs count==" << nr_pairs  << "\n";
	for (const auto& pair : pairs) 
	{
		std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
        }
	
	
	// Using a map to store adjacency list
    	std::map<int, std::set<int>> graph;

    	// Build the graph
    	for (auto& edge : pairs) 
    	{
		// Add each node to the adjacency list of the other node
		graph[edge.first].insert(edge.second);
		graph[edge.second].insert(edge.first);  // Uncomment this if the graph is undirected
	}

       // Print the graph
    	for (auto& node : graph) 
    	{
    		std::cout << "Node " << node.first << " has edges with: ";
		for (int neighbor : node.second) 
		{
		    std::cout << neighbor << " ";
		}
		std::cout << std::endl;
   	 }
   	 
   	 // Visited nodes set
	    std::set<int> visited;
	    int num_components = 0;

   	 // Finding all components
	    for (auto& node : graph) 
	    {
		if (visited.find(node.first) == visited.end()) 
		{
		    // Perform DFS from this node
		    DFS(node.first, graph, visited);
		    num_components++;  // Each DFS call represents a new component
		}
	    }

	 // Output the number of disconnected components
	 std::cout << "The graph has " << num_components << " disconnected component(s)." << std::endl;
	
	
	
	//perform pairwise registration
	coarseRgistration(keypoint_clouds, keypoint_clouds_feature, clouds, pairs_best_count, candidate_matches);
	
	globalFineRegistration(candidate_matches, clouds, poses, num_of_subtrees);

	cout << "666666666666666nr_scans==" << nr_scans << "\n";
		
		
		
	loop=1;
	Hierarchical_block_poses.push_back(block);
	//check how many scans are left
	
	

	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> past = std::chrono::duration_cast<std::chrono::duration<double>>(end - begin);
	
	int min = (past.count()-pretmp.count()-pretmp1.count())/ 60;
	double sec = past.count()-pretmp.count()-pretmp1.count()- 60 * min;
	cout << "total cost time:" << min << "min" << sec << "sec\n";
	
}

